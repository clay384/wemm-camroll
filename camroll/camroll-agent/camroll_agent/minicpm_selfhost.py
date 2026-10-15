"""Self-hosted Q6 transport; shares accounting/validation, never cloud defaults."""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
import time
from urllib.parse import urlsplit
import uuid

from .budget import BudgetLedger, RequestJournal
from .modelbest import ModelBestClient, _rejection

MODEL = "minicpm-v4.7-35b-a3b-q6-thinking"
PROTOCOL = "minicpm47-q6-thinking-m4-off-v1"


def pricing_evidence(value):
    if (not isinstance(value, dict) or value.get("model") != MODEL
            or value.get("provider") != "llamacpp-selfhost"
            or value.get("input_rate") != 0 or value.get("output_rate") != 0
            or value.get("cost_scope") != "api_only"
            or value.get("infrastructure_cost_cny") is not None
            or not value.get("deployment_manifest_sha256") or not value.get("observed_at")):
        raise ValueError("Self-hosted API-only cost evidence and deployment binding required")
    return dict(value)


class MiniCPMSelfHostedClient(ModelBestClient):
    model = MODEL
    provider = "llamacpp-selfhost"
    reasoning_field = "reasoning_content"

    def __init__(self, *, pricing_evidence, sdk=None, ledger=None, base_url=None,
                 api_key=None, sleep=time.sleep, timeout_s=600, request_namespace=None,
                 request_journal_path=None, strict_response_model=True,
                 decision_mode="native_tools", json_mode="json_schema",
                 request_profile="selfhost-q6-v1", thinking_mode="server-on",
                 compatibility_mode="selfhost-q6-v1", expected_limit_cny=600):
        self.pricing_evidence = globals()["pricing_evidence"](pricing_evidence)
        if (timeout_s != 600 or strict_response_model is not True or expected_limit_cny != 600
                or request_profile != "selfhost-q6-v1" or thinking_mode != "server-on"
                or compatibility_mode != "selfhost-q6-v1"
                or decision_mode not in ("native_tools", "json_decision")
                or json_mode not in ("json_schema", "json_object", "prompt")):
            raise ValueError("Self-hosted Q6 protocol settings differ")
        self.base_url = (base_url or os.environ.get("CAMROLL_MINICPM_BASE_URL", "")).rstrip("/")
        url = urlsplit(self.base_url)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
                or url.path != "/v1" or url.username or url.password or url.query or url.fragment):
            raise ValueError("Self-hosted API must use the explicit loopback SSH tunnel")
        self.model, self.protocol = MODEL, PROTOCOL
        self.decision_mode, self.json_mode = decision_mode, json_mode
        self.request_profile, self.thinking_mode = request_profile, thinking_mode
        self.compatibility_mode = compatibility_mode
        self.context_token_bound, self.input_token_bound = 262144, 262144 - 8192
        self.timeout_s, self.strict_response_model, self._sleep = timeout_s, True, sleep
        self.pricing_version = "selfhost-api-only:" + hashlib.sha256(json.dumps(self.pricing_evidence, sort_keys=True).encode()).hexdigest()
        self._observer_lock, self._observers = threading.RLock(), []
        self._json_metric_buffer = contextvars.ContextVar("selfhost_json_metrics", default=None)
        self.ledger = (ledger.with_expected_limit(600) if ledger is not None else
                       BudgetLedger(os.environ.get("CAMROLL_BUDGET_DB", "~/.local/state/camroll/budget.sqlite3"),
                                    limit_cny=600, expected_limit_cny=600))
        self.request_namespace, self._request_ordinal = request_namespace or uuid.uuid4().hex, 0
        self.journal = RequestJournal(request_journal_path or self.ledger.path.with_name("selfhost_requests.sqlite"))
        self._capability_key = "selfhost:" + hashlib.sha256((self.base_url + self.model).encode()).hexdigest()
        if sdk is None:
            from openai import OpenAI
            import httpx
            sdk = OpenAI(api_key="local-no-auth", base_url=self.base_url, max_retries=0,
                         timeout=timeout_s, http_client=httpx.Client(trust_env=False))
        self.sdk, self._error_secrets = sdk, ()

    def capabilities(self):
        return {"provider": self.provider, "model": self.model, "protocol_version": self.protocol,
                "context_token_bound": 262144, "max_output_tokens": 8192,
                "requested_max_output_tokens": 8192, "decision_mode": self.decision_mode,
                "json_mode": self.json_mode, "enable_thinking": True, "thinking_mode": "server-on",
                "stream": False, "temperature": 0, "pricing_version": self.pricing_version,
                "request_profile": self.request_profile, "request_timeout_s": 600}

    @staticmethod
    def _parse_json(content):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        return json.loads(content, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))

    def _generation_parameters(self, max_tokens):
        return dict(max_tokens=max_tokens, temperature=0, stream=False)

    @staticmethod
    def _classify_error(exc):
        body = getattr(exc, "body", {})
        error = body.get("error", body) if isinstance(body, dict) else {}
        if (getattr(exc, "status_code", None) in (400, 413, 422)
                and isinstance(error, dict) and error.get("type") == "exceed_context_size_error"):
            return "context_limit", True, False
        return _rejection(exc)
