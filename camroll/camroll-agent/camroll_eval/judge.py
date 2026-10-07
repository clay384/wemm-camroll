"""Independent Judge transport using frozen PhotoBench prompts and scoring rules."""
from __future__ import annotations

import json
import contextlib
import contextvars
import hashlib
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from camroll_agent.budget import BudgetLedger, RequestReplayError, RequestUncertain
from .common import read_json, read_secret, object_hash, now_iso
from .compat import JudgeProtocol


class Judge(JudgeProtocol):
    def __init__(self, *, transport=None, cancel=None, on_call=None, model=None, url=None,
                 retries=6, interval=.5, prompts=None, ledger=None, request_namespace=None,
                 request_journal_path=None, expected_limit_cny=None):
        self.judge_model = model or os.getenv("CAMROLL_EVAL_JUDGE_MODEL", "doubao-seed-2.0-lite")
        self.judge_url = url or os.getenv("CAMROLL_EVAL_JUDGE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
        self.prompts = prompts or read_json(Path(__file__).with_name("judge_prompts.json"))
        self.judge_system_prompt = self.prompts["answer_quality"]
        self.task_judge_system_prompt = self.prompts["task_decision"]
        self.evidence_judge_system_prompt = self.prompts["evidence"]
        self._cancel = cancel or threading.Event()
        self._rate_lock = threading.Lock()
        self._next_request = 0.
        self.on_call = on_call
        self.retries, self.interval = retries, interval
        self.transport = transport
        self._key = None
        self._qwen = None
        self.ledger = ledger
        if ledger is not None and expected_limit_cny is not None:
            self.ledger = ledger.with_expected_limit(expected_limit_cny)
        self.stop_reason = None
        self.request_namespace = request_namespace
        self._request_context = contextvars.ContextVar('judge_request_context', default=None)
        self.journal = None
        if request_journal_path:
            from camroll_agent.budget import RequestJournal
            self.journal = RequestJournal(request_journal_path)
        self.input_rate = max(1.8, float(os.getenv("CAMROLL_EVAL_JUDGE_INPUT_CNY_PER_MILLION", "1.8")))
        self.output_rate = max(10.8, float(os.getenv("CAMROLL_EVAL_JUDGE_OUTPUT_CNY_PER_MILLION", "10.8")))
        if transport is None:
            if self.judge_model == "qwen3.7-flash":
                from camroll_agent.qwen import QwenClient
                self._qwen = QwenClient()  # Uses the same cumulative budget ledger.
                self.judge_url = self._qwen.base_url
            else:
                self._key = read_secret("CAMROLL_EVAL_JUDGE_API_KEY")
                budget_path = os.environ.get("CAMROLL_BUDGET_DB")
                if not budget_path or not Path(budget_path).expanduser().is_file():
                    raise ValueError("existing CAMROLL_BUDGET_DB is required for Judge accounting")
                self.ledger = self.ledger or BudgetLedger(budget_path, expected_limit_cny or 300.,
                                                         expected_limit_cny=expected_limit_cny)

    @contextlib.contextmanager
    def request_context(self, namespace):
        token = self._request_context.set({'namespace': namespace, 'ordinal': 0})
        try:
            yield
        finally:
            self._request_context.reset(token)

    def protocol(self):
        return {"model": self.judge_model, "url": self.judge_url,
                "prompt_hashes": {key: object_hash(value) for key, value in self.prompts.items()},
                "temperature": 0, "evidence_judge_enabled": False,
                "billing_mode": "subscription_equivalent_estimate" if not self._qwen else "qwen_usage_estimate",
                "input_rate_cny_per_million": self.input_rate,
                "output_rate_cny_per_million": self.output_rate,
                "pricing_source": "https://www.volcengine.com/docs/84458/1585097?lang=zh&redirect=1",
                "pricing_observed_date": "2026-09-16", "pricing_policy": "highest_context_tier_no_cache_discount",
                "retry_attempts": self.retries, "minimum_interval_seconds": self.interval}

    def _wait(self):
        with self._rate_lock:
            now = time.monotonic()
            delay = max(0., self._next_request - now)
            self._next_request = max(now, self._next_request) + self.interval
        if self._cancel.wait(delay):
            raise RuntimeError("judge_cancelled")

    def _send(self, payload, timeout):
        if self.transport:
            return self.transport(payload)
        if self._qwen:
            result = self._qwen.chat(payload["messages"], role="benchmark_judge",
                                     max_tokens=payload["max_tokens"])
            return {"choices": [{"message": result}], "usage": result.get("_usage", {})}
        request = urllib.request.Request(self._judge_chat_url(),
            data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def _judge_request(self, payload, timeout=180):
        # QwenClient already applies bounded retries and reservations; do not
        # multiply those retries in a second layer.
        attempts = 1 if self._qwen else self.retries
        maximum = int(payload["max_tokens"])
        upper_bound = len(json.dumps(payload["messages"], ensure_ascii=False).encode()) + maximum + 1024
        if upper_bound > 262144:
            raise ValueError("judge_context_limit")
        context = self._request_context.get()
        if context:
            context['ordinal'] += 1
            identity = f"{self.request_namespace}:judge:{context['namespace']}:{context['ordinal']}"
        else:
            identity = f"{self.request_namespace}:judge:{uuid.uuid4().hex}"
        payload_hash = object_hash(payload)
        for attempt in range(1, attempts + 1):
            request_id = identity + ':' + str(attempt)
            self._wait()
            began = time.monotonic()
            event = {"started_at": now_iso(), "model": self.judge_model, "attempt": attempt,
                     "role": "judge", "input_max_tokens": payload["max_tokens"]}
            call_id = None
            replayed = False
            result = None
            if self.ledger and not self._qwen:
                try:
                    def reserve():
                        return self.ledger.reserve((262144 * self.input_rate + maximum * self.output_rate) / 1_000_000,
                        {"role": "benchmark_judge", "model": self.judge_model, "max_tokens": maximum,
                         "pricing_version": "doubao-lite-20260916-subscription-equivalent-highest-tier",
                         "input_rate": self.input_rate, "output_rate": self.output_rate})
                    if self.journal:
                        claim = self.journal.claim(request_id, payload_hash, reserve)
                        call_id = claim['call_id']
                        replayed = claim.get('replayed', False)
                        result = claim['response']
                    else:
                        call_id = reserve()
                except RequestReplayError as exc:
                    if exc.code == 'http_429' and attempt < attempts:
                        continue
                    raise
                except Exception as exc:
                    if type(exc).__name__ in {'BudgetExceeded', 'BudgetLimitChanged', 'RequestUncertain'}:
                        self.stop_reason = getattr(exc, 'code', 'budget_exhausted')
                        self._cancel.set()
                    raise
            try:
                if not replayed:
                    result = self._send(payload, timeout)
                if not isinstance(result, dict) or not result.get("choices"):
                    raise ValueError("judge_invalid_envelope")
                if self.journal and not replayed:
                    self.journal.complete(request_id, result)
                usage = result.get("usage") or {}
                prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
                known = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (prompt, completion))
                cost = (prompt * self.input_rate + completion * self.output_rate) / 1_000_000 if known and not self._qwen else None
                if call_id:
                    self.ledger.finish(call_id, cost,
                        {"prompt_tokens": prompt if known else None, "completion_tokens": completion if known else None,
                         "total_tokens": prompt + completion if known else None,
                         "response_model": result.get("model"), "usage_missing": not known,
                         "elapsed_s": time.monotonic() - began})
                if not known and call_id:
                    if self.journal:
                        self.journal.fail(request_id, 'usage_missing', uncertain=True)
                    self.stop_reason = 'request_uncertain'
                    self._cancel.set()
                    raise RequestUncertain('Judge usage is missing; the full reservation is retained')
                event.update(status="complete", usage=result.get("usage") or {},
                             response_model=result.get("model"),
                             duration_ms=round((time.monotonic() - began) * 1000, 3),
                             budget_call_id=call_id, estimated_cost_cny=0 if replayed else cost, replayed=replayed,
                             billing_mode="subscription_equivalent_estimate" if not self._qwen else "qwen_usage_estimate")
                if self.on_call:
                    self.on_call(event)
                return result, attempt
            except Exception as exc:
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if call_id:
                    rejected = code in {400, 401, 403, 404, 422, 429}
                    self.ledger.finish(call_id, 0 if rejected else None,
                                       {"error_code": "http_" + str(code) if isinstance(code, int) else "transport_error",
                                        "elapsed_s": time.monotonic() - began}, rejected=rejected)
                    if self.journal:
                        self.journal.fail(request_id, 'http_' + str(code) if rejected else 'transport_uncertain',
                                          uncertain=not rejected)
                    if not rejected:
                        self.stop_reason = 'request_uncertain'
                        self._cancel.set()
                event.update(status="error", error_type=type(exc).__name__, http_status=code if isinstance(code, int) else None,
                             duration_ms=round((time.monotonic() - began) * 1000, 3), estimated_cost_cny=None)
                if self.on_call:
                    self.on_call(event)
                # Never return provider exception bodies (may include headers).
                if code in {400, 401, 403, 404, 422} or attempt == attempts or self._cancel.is_set():
                    raise RuntimeError("judge_transport_" + type(exc).__name__) from None
                if self._cancel.wait(min(60., 5. * 2 ** (attempt - 1))):
                    raise RuntimeError("judge_cancelled") from None

    def score_item(self, item, row):
        """Every annotated turn is judged; final-turn quality counts once per QA."""
        records = item.get("runtime_turns") or []
        definitions = row.get("conversation") or []
        for index, record in enumerate(records):
            if self._cancel.is_set():
                break
            if (record.get("judge") or {}).get("score") in {0, 1, 2} and (record.get("task_judge") or {}).get("actual_action") in {"answer", "clarify", "refuse", "none"}:
                continue
            expected = record.get("expected_action")
            definition = definitions[index] if index < len(definitions) else {}
            reference = definition.get("reference_answer") or (
                "应要求用户提供至少一个具体、可用于后续检索的锚点；询问形式不限。"
                if expected == "clarify" else row.get("answer", ""))
            context = records[:index + 1]
            record["task_judge"] = self._judge_task_action(record["message"], record.get("answer", ""), expected,
                record.get("agent_status"), record.get("termination_reason", ""),
                task_type=row.get("task_type"), question_type=row.get("question_type"),
                answerability=row.get("answerability"), reference=reference, conversation=context)
            record["judge"] = self._judge(record["message"], reference, record.get("answer", ""),
                expected_action=expected, task_type=row.get("task_type"), question_type=row.get("question_type"),
                answerability=row.get("answerability"), conversation=context)
            if record["judge"].get("score") not in {0, 1, 2}:
                record["judge"]["judge_status"] = "failed"
            if record["task_judge"].get("actual_action") not in {"answer", "clarify", "refuse", "none"}:
                record["task_judge"]["judge_status"] = "failed"
            record["evidence_judge"] = {"score": None, "reason": "disabled"}
        item["task_judges"] = [record.get("task_judge") or {"expected_action": record.get("expected_action"), "correct": None} for record in records]
        item["task_judge"] = item["task_judges"][-1] if records else {"correct": None}
        final_turn_executed = len(records) == (len(definitions) or 1)
        item["final_turn_executed"] = final_turn_executed
        item["judge"] = ((records[-1].get("judge") or {"score": None, "reason": "cancelled"})
                         if final_turn_executed else {"score": None, "reason": "final_turn_not_executed", "judge_status": "incomplete"})
        item["judge_status"] = "failed" if any(
            (record.get(key) or {}).get("judge_status") == "failed" for record in records
            for key in ("task_judge", "judge")) else "completed"
        if self._cancel.is_set() and any((record.get("judge") or {}).get("score") not in {0, 1, 2} for record in records):
            item["judge_status"] = "cancelled"
        elif not final_turn_executed:
            item["judge_status"] = "incomplete"
        return item
