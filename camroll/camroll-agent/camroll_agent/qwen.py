"""Qwen3.7-Flash cloud adapter with bounded retries and shared budget accounting."""
from __future__ import annotations

import base64
import contextlib
import contextvars
import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .budget import BudgetExceeded, BudgetLedger

MODEL = "qwen3.7-flash"
# Full-context reservation, deliberately independent of token estimation and
# image compression. Local preflight bounds input below the model context.
CONTEXT_TOKEN_BOUND = 1_048_576
MAX_OUTPUT_TOKENS = 32_768
PRICING_VERSION = "qwen3.7-flash-cn-2026-09-15"


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, code: str = "provider_error", status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class RequestLimitExceeded(ProviderError):
    def __init__(self):
        super().__init__("The per-question cloud request limit was reached", code="request_limit")


class DeadlineExceeded(ProviderError):
    def __init__(self):
        super().__init__("The per-question deadline was reached", code="timeout")


class RequestScope:
    def __init__(self, max_requests: int, deadline_s: float):
        self.max_requests = max_requests
        self.started = time.monotonic()
        self.deadline = self.started + deadline_s
        self.requests = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.estimated_cny = 0.0
        self.reserved_cny = 0.0
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise DeadlineExceeded()
        if self.requests >= self.max_requests:
            raise RequestLimitExceeded()

    def start_request(self) -> None:
        with self._lock:
            self.check()
            self.requests += 1

    def record(self, usage: dict, cost: float | None, reservation: float) -> None:
        with self._lock:
            for key in self.usage:
                self.usage[key] += int(usage.get(key, 0) or 0)
            self.estimated_cny += cost or 0.0
            if cost is None:
                self.reserved_cny += reservation

    def summary(self) -> dict:
        with self._lock:
            return {"requests": self.requests, "max_requests": self.max_requests,
                    "elapsed_s": round(time.monotonic() - self.started, 4),
                    "estimated_cny": round(self.estimated_cny, 8),
                    "reserved_cny": round(self.reserved_cny, 8), **self.usage}

    def record_call(self, call: dict) -> None:
        with self._lock:
            self.calls.append(call)


_scope: contextvars.ContextVar[RequestScope | None] = contextvars.ContextVar("camroll_qwen_scope", default=None)


def _trace_payload(value):
    """Keep full text/tool I/O, but never duplicate private image bytes in traces."""
    if isinstance(value, dict):
        return {k: _trace_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_trace_payload(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_payload_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "encoded_bytes": len(value)}
    return value


def image_messages(prompt: str, image_paths: list[str | Path]) -> list[dict]:
    """Construct a bounded JPEG multimodal message from trusted asset paths."""
    from PIL import Image, ImageOps

    content: list[dict] = [{"type": "text", "text": prompt}]
    for i, path in enumerate(image_paths):
        with Image.open(path) as original:
            picture = ImageOps.exif_transpose(original).convert("RGB")
            picture.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            picture.save(output, format="JPEG", quality=88)
        data = base64.b64encode(output.getvalue()).decode("ascii")
        content.append({"type": "text", "text": "Image {} (same order as supplied IDs):".format(i + 1)})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}})
    return [{"role": "user", "content": content}]


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return vars(value) if hasattr(value, "__dict__") else {}


def _rates(prompt_tokens: int) -> tuple[float, float]:
    if prompt_tokens <= 32_768:
        return 0.2, 0.8
    if prompt_tokens <= 262_144:
        return 0.6, 2.4
    return 1.2, 4.8


def _usage_info(value: Any) -> tuple[dict, float | None]:
    raw = _as_dict(value)
    if not all(isinstance(raw.get(k), int) and raw[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
        return {"usage_missing": True}, None
    prompt, completion = raw["prompt_tokens"], raw["completion_tokens"]
    input_rate, output_rate = _rates(prompt)
    details = _as_dict(raw.get("prompt_tokens_details", {}))
    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
             "total_tokens": prompt + completion,
             "cached_tokens": int(details.get("cached_tokens", 0) or 0),
             "input_rate": input_rate, "output_rate": output_rate}
    # Cache discounts are never assumed, keeping the budget conservative.
    return usage, (prompt * input_rate + completion * output_rate) / 1_000_000


class QwenClient:
    model = MODEL

    def __init__(self, *, sdk: Any = None, ledger: BudgetLedger | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 sleep: Any = time.sleep, timeout_s: float = 90.0):
        self.base_url = base_url or os.environ.get("QWEN_BASE_URL")
        if not self.base_url:
            if sdk is None:
                raise ProviderError("QWEN_BASE_URL must specify the authorized business workspace endpoint", code="missing_endpoint")
            self.base_url = "https://injected-sdk.invalid/v1"
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._observer_lock = threading.RLock()
        self._observers = []
        ledger_path = os.environ.get("CAMROLL_BUDGET_DB", "~/.local/state/camroll/budget.sqlite3")
        self.ledger = ledger or BudgetLedger(ledger_path, min(300.0, float(os.environ.get("CAMROLL_BUDGET_CNY", "300"))))
        if sdk is None:
            from openai import OpenAI

            key = api_key or os.environ.get("DASHSCOPE_API_KEY")
            if not key:
                raise ProviderError("DASHSCOPE_API_KEY is required", code="missing_credentials")
            sdk = OpenAI(api_key=key, base_url=self.base_url, max_retries=0, timeout=timeout_s)
        self.sdk = sdk
        self._capability_key = "json_schema:" + hashlib.sha256((self.base_url + MODEL).encode()).hexdigest()

    @contextlib.contextmanager
    def request_scope(self, max_requests: int = 50, deadline_s: float = 180):
        if max_requests < 1 or deadline_s <= 0:
            raise ValueError("Request scope limits must be positive")
        existing = _scope.get()
        if existing is not None:
            # Multiple role adapters inside one question share one allowance.
            yield existing
            return
        scope = RequestScope(max_requests, deadline_s)
        token = _scope.set(scope)
        try:
            yield scope
        finally:
            _scope.reset(token)

    def usage_summary(self) -> dict:
        return self.ledger.summary()

    @contextlib.contextmanager
    def capture_calls(self, callback):
        """Observe this client's build calls across worker threads, without budgets.

        Use one client per build/run. The callback must handle concurrent calls;
        it receives a detached JSON snapshot with image bytes removed.
        """
        with self._observer_lock:
            self._observers.append(callback)
        try:
            yield
        finally:
            with self._observer_lock:
                self._observers.remove(callback)

    def _emit_metric(self, metric):
        with self._observer_lock:
            observers = list(self._observers)
        for callback in observers:
            callback(json.loads(json.dumps(metric, ensure_ascii=False)))

    def capabilities(self) -> dict:
        return {"json_schema": self.ledger.capability(self._capability_key), "model": MODEL}

    @staticmethod
    def _preflight(messages: list[dict], tools: list | None, response_format: dict | None, max_tokens: int):
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
            raise ProviderError("max_tokens must be between 1 and 32768", code="invalid_request")
        # UTF-8 bytes upper-bound text tokens; image token budget is conservative
        # for provider-bound images with a maximum 2048-pixel long edge.
        token_bound = 1024 + max_tokens
        if tools:
            token_bound += len(json.dumps(tools, ensure_ascii=False).encode("utf-8"))
        if response_format:
            token_bound += len(json.dumps(response_format, ensure_ascii=False).encode("utf-8"))
        for message in messages:
            copy = dict(message)
            body = copy.pop("content", "")
            token_bound += len(json.dumps(copy, ensure_ascii=False).encode("utf-8"))
            if isinstance(body, list):
                for item in body:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if not url.startswith("data:image/jpeg;base64,"):
                            raise ProviderError("Images must use image_messages() JPEG payloads", code="invalid_image")
                        from PIL import Image
                        try:
                            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))) as picture:
                                if max(picture.size) > 2048 or picture.format != "JPEG":
                                    raise ValueError("unbounded image")
                        except Exception as exc:
                            raise ProviderError("Invalid or oversized image payload", code="invalid_image") from None
                        token_bound += 32_768
                    else:
                        token_bound += len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            else:
                token_bound += len(str(body or "").encode("utf-8"))
        if token_bound > CONTEXT_TOKEN_BOUND:
            raise ProviderError("Request exceeds the conservative context limit", code="context_limit")

    def chat(self, messages: list[dict], tools: list | None = None, *, role: str = "agent",
             response_format: dict | None = None, max_tokens: int = 2048,
             tool_choice: str | dict = "auto", stream: bool = False) -> dict:
        self._preflight(messages, tools, response_format, max_tokens)
        reservation = (CONTEXT_TOKEN_BOUND * 1.2 + max_tokens * 4.8) / 1_000_000
        body: dict = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                      "temperature": 0, "extra_body": {"enable_thinking": False}}
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        if response_format:
            body["response_format"] = response_format
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        for attempt in range(3):
            scope = _scope.get()
            if scope:
                scope.check()
            metadata = {"role": role, "model": MODEL, "prompt_version": "camroll-qwen-v1",
                        "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False,
                        "pricing_version": PRICING_VERSION, "stream": stream}
            call_id = self.ledger.reserve(reservation, metadata)
            try:
                if scope:
                    scope.start_request()
            except Exception:
                self.ledger.finish(call_id, 0, {"error_code": "scope_limit"}, rejected=True)
                raise
            timeout = self.timeout_s if scope is None else min(self.timeout_s, max(0.001, scope.deadline - time.monotonic()))
            started = time.monotonic()
            metric = {"call_id": call_id, "call_type": role, "model": MODEL, "attempt": attempt + 1,
                      "started_at_epoch": time.time(), "streaming": stream, "ttft_ms": None,
                      "parse_status": None, "input": _trace_payload(body), "status": "running"}
            if scope:
                scope.record_call(metric)
            try:
                response = self.sdk.chat.completions.create(**body, timeout=timeout)
                if stream:
                    response = self._collect_stream(response, started=started)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                name = type(exc).__name__.lower()
                # Exception content is only used locally to classify unsupported
                # capabilities. It must not enter logs, errors, or budget data.
                error_text = str(exc).lower()
                unsupported_schema = (status == 400 and response_format and
                                      response_format.get("type") == "json_schema" and
                                      "json_schema" in error_text and any(x in error_text for x in
                                      ("not support", "unsupported", "not available", "must be one of", "only support")))
                rejected = status in (400, 401, 403, 404, 422, 429)
                code = "unsupported_schema" if unsupported_schema else ("http_{}".format(status) if status else "transport_error")
                metric.update(status="error", error_code=code, duration_ms=(time.monotonic()-started)*1000,
                              finished_at_epoch=time.time())
                self.ledger.finish(call_id, 0 if rejected else None,
                                   {"error_code": code, "elapsed_s": time.monotonic() - started}, rejected=rejected)
                if scope:
                    scope.record({}, 0 if rejected else None, reservation)
                self._emit_metric(metric)
                if isinstance(exc, DeadlineExceeded):
                    raise
                retryable = status == 429 or (isinstance(status, int) and 500 <= status <= 599) or any(
                    word in name for word in ("timeout", "connection"))
                if not retryable or attempt == 2:
                    raise ProviderError("Qwen request failed ({})".format(code), code=code, status_code=status) from None
                wait = min(4.0, 0.5 * (2 ** attempt))
                if scope and time.monotonic() + wait >= scope.deadline:
                    raise DeadlineExceeded() from None
                self._sleep(wait)
                continue
            raw = _as_dict(response)
            usage, cost = _usage_info(raw.get("usage"))
            response_model = raw.get("model", MODEL)
            duration_ms = (time.monotonic() - started) * 1000
            ttft_ms = raw.get("_ttft_ms") if stream else None
            generation_ms = duration_ms - ttft_ms if ttft_ms is not None else None
            metric.update(status="success", model=response_model, duration_ms=duration_ms,
                          finished_at_epoch=time.time(), ttft_ms=ttft_ms,
                          prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                          tokens_per_second=(usage.get("completion_tokens", 0)*1000/generation_ms
                                             if generation_ms and generation_ms > 0 and not usage.get("usage_missing") else None),
                          estimated_cny=cost, output=_trace_payload(raw.get("choices", [])))
            self.ledger.finish(call_id, cost, {**usage, "response_model": response_model,
                                               "elapsed_s": time.monotonic() - started})
            if scope:
                scope.record(usage, cost, reservation)
                if time.monotonic() >= scope.deadline:
                    metric.update(status="late_response", error_code="deadline_exceeded")
                    self._emit_metric(metric)
                    raise DeadlineExceeded()
            choices = raw.get("choices", [])
            if not choices:
                metric.update(status="error", error_code="empty_response")
                self._emit_metric(metric)
                raise ProviderError("Qwen returned no assistant choice", code="empty_response")
            choice = _as_dict(choices[0])
            message = _as_dict(choice.get("message"))
            if message.get("refusal"):
                metric.update(status="error", error_code="refusal")
                self._emit_metric(metric)
                raise ProviderError("Qwen declined this request", code="refusal")
            if message.get("role", "assistant") != "assistant":
                metric.update(status="error", error_code="invalid_message")
                self._emit_metric(metric)
                raise ProviderError("Qwen returned an invalid message role", code="invalid_message")
            normalized = {"role": "assistant", "content": message.get("content"),
                          "_usage": {**usage, "estimated_cny": cost, "model": response_model},
                          "_finish_reason": choice.get("finish_reason"), "_metric": metric}
            if message.get("tool_calls"):
                calls = []
                for call in message["tool_calls"]:
                    item = _as_dict(call)
                    function = _as_dict(item.get("function"))
                    calls.append({"id": item.get("id"), "type": item.get("type", "function"),
                                  "function": {"name": function.get("name"), "arguments": function.get("arguments")}})
                normalized["tool_calls"] = calls
            if not isinstance(normalized["content"], (str, type(None))):
                metric.update(status="error", error_code="invalid_message")
                self._emit_metric(metric)
                raise ProviderError("Qwen returned unsupported assistant content", code="invalid_message")
            if not normalized.get("tool_calls") and not normalized.get("content"):
                metric.update(status="error", error_code="empty_response")
                self._emit_metric(metric)
                raise ProviderError("Qwen returned empty assistant content", code="empty_response")
            self._emit_metric(metric)
            return normalized
        raise AssertionError("unreachable")

    @staticmethod
    def _collect_stream(response: Any, started: float | None = None) -> dict:
        started = time.monotonic() if started is None else started
        ttft_ms = None
        content: list[str] = []
        calls: dict[int, dict] = {}
        usage = None
        model = MODEL
        finish_reason = None
        try:
            for chunk in response:
                scope = _scope.get()
                if scope and time.monotonic() >= scope.deadline:
                    raise DeadlineExceeded()
                raw = _as_dict(chunk)
                model = raw.get("model", model)
                if raw.get("usage"):
                    usage = raw["usage"]
                for raw_choice in raw.get("choices", []):
                    choice = _as_dict(raw_choice)
                    if choice.get("index", 0) != 0:
                        continue
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = _as_dict(choice.get("delta"))
                    meaningful_call = any(any(_as_dict(_as_dict(c).get("function")).get(k)
                                              for k in ("name", "arguments"))
                                          for c in delta.get("tool_calls", []))
                    if ttft_ms is None and (delta.get("content") or meaningful_call):
                        ttft_ms = (time.monotonic() - started) * 1000
                    if delta.get("content"):
                        content.append(delta["content"])
                    for raw_call in delta.get("tool_calls", []):
                        call = _as_dict(raw_call)
                        index = call.get("index", 0)
                        target = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if call.get("id"):
                            target["id"] = call["id"]
                        function = _as_dict(call.get("function"))
                        for key in ("name", "arguments"):
                            target["function"][key] += function.get(key) or ""
        finally:
            if hasattr(response, "close"):
                response.close()
        return {"model": model, "usage": usage, "_ttft_ms": ttft_ms,
                "choices": [{"finish_reason": finish_reason, "message": {
                    "role": "assistant", "content": "".join(content) or None,
                    "tool_calls": [calls[i] for i in sorted(calls)]}}]}

    def generate_json(self, prompt: str, image_paths: list[str | Path], schema: dict, *,
                      role: str = "caption", max_attempts: int = 3) -> dict:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError

        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError:
            raise ProviderError("Invalid local output schema", code="invalid_schema") from None
        validator = Draft202012Validator(schema)
        messages = image_messages(prompt + "\nReturn exactly one JSON object matching the supplied schema.\n" +
                                  json.dumps(schema, ensure_ascii=False), image_paths)
        messages.insert(0, {"role": "system", "content": "You are a careful assistant. Follow the requested JSON schema; do not invent facts."})
        error_detail = "invalid JSON object"
        for attempt in range(max_attempts):
            schema_enabled = self.ledger.capability(self._capability_key) is not False
            response_format = ({"type": "json_schema", "json_schema": {"name": "camroll_output", "strict": True, "schema": schema}}
                               if schema_enabled else {"type": "json_object"})
            try:
                try:
                    result = self.chat(messages, role=role, response_format=response_format)
                    if schema_enabled:
                        self.ledger.set_capability(self._capability_key, True)
                except ProviderError as exc:
                    if exc.code != "unsupported_schema":
                        raise
                    self.ledger.set_capability(self._capability_key, False)
                    result = self.chat(messages, role=role, response_format={"type": "json_object"})
                content = result.get("content")
                parsed = json.loads(content) if isinstance(content, str) else None
                if not isinstance(parsed, dict):
                    error_detail = "Top-level value must be a JSON object"
                else:
                    errors = sorted(validator.iter_errors(parsed), key=lambda error: str(list(error.path)))
                    if not errors:
                        result.get("_metric", {})["parse_status"] = "valid"
                        return parsed
                    # Validator messages may contain the entire rejected value;
                    # pass only paths and validation keywords back to the model.
                    error_detail = "; ".join("{}: {}".format(".".join(map(str, e.path)) or "$", e.validator) for e in errors[:5])
                result.get("_metric", {})["parse_status"] = "invalid"
                messages.append({"role": "assistant", "content": content or "null"})
            except json.JSONDecodeError:
                result.get("_metric", {})["parse_status"] = "invalid"
                error_detail = "Response was not valid JSON"
                messages.append({"role": "assistant", "content": result.get("content") or "null"})
            except ProviderError as exc:
                if exc.code not in ("empty_response", "invalid_message"):
                    raise
                error_detail = "Response must contain a JSON object"
            if attempt + 1 < max_attempts:
                messages.append({"role": "user", "content": "Correct the previous response. Validation failed: " + error_detail + ". Return only a valid JSON object."})
        raise ProviderError("Structured output failed validation after {} attempts ({})".format(max_attempts, error_detail), code="invalid_json")

    def generate(self, prompt: str, image_paths: list[str | Path]) -> str:
        return self.chat(image_messages(prompt, image_paths), role="view")["content"] or ""
