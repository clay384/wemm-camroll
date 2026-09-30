from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from camroll_agent import agent as agent_module
from camroll_agent import cli, library, pipeline, qwen, runtime_clients
from camroll_agent.agent import Agent
from camroll_agent.experiment import Q38


class ProfileClient:
    model = "qwen3.8-flash"

    def __init__(self, **kwargs):
        self.options = kwargs
        self.scope_options = None
        self.model = kwargs.get("model", self.model)

    @contextmanager
    def request_scope(self, **kwargs):
        self.scope_options = kwargs
        yield SimpleNamespace(summary=lambda: {})

    def chat(self, messages, **kwargs):
        return {"role": "assistant", "content": "没有足够证据。"}


def profile_library():
    return SimpleNamespace(experiment=Q38, cache_namespace="q38_album", config="M4", version="frozen", metadata={})


def test_direct_agent_infers_snapshot_model_mode_deadline_before_client_creation(monkeypatch, tmp_path):
    seen = []

    def load(path, **kwargs):
        seen.append(kwargs)
        return profile_library()

    monkeypatch.setattr(library, "Library", load)
    monkeypatch.setattr(qwen, "QwenClient", ProfileClient)
    value = Agent(str(tmp_path), max_steps=1)
    assert seen == [{"embedding_client": None, "config": None}]
    assert value.config == "M4"
    assert value.deadline_s == 600
    assert value.llm is value.vlm
    assert value.llm.options == {"model": "qwen3.8-flash", "expected_limit_cny": 600,
                                 "request_namespace": "q38_album", "request_journal_path": tmp_path / ".request-journal.sqlite"}
    value.ask("问题")
    assert value.llm.scope_options == {"max_requests": 50, "deadline_s": 600}


@pytest.mark.parametrize("options", [{"llm_model": "qwen3.7-flash"}, {"config": "M1"},
                                     {"deadline_s": 180}, {"llm": SimpleNamespace(model="qwen3.7-flash")},
                                     {"vlm": SimpleNamespace(model="qwen3.7-flash")}])
def test_agent_rejects_generation_profile_mismatch_before_creating_client(monkeypatch, options):
    def forbidden(**kwargs):
        raise AssertionError("Client must not be created for a rejected profile")

    monkeypatch.setattr(qwen, "QwenClient", forbidden)
    with pytest.raises(ValueError):
        Agent("unused", library=profile_library(), **options)


def test_legacy_agent_retains_m1_and_180_second_defaults():
    value = Agent("unused", llm=ProfileClient(), library=SimpleNamespace(version="legacy", metadata={}))
    assert value.config == "M1"
    assert value.deadline_s == 180


def test_local_scope_uses_full_600_seconds(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(agent_module.time, "monotonic", lambda: clock[0])
    scope = agent_module._LocalScope(600)
    clock[0] = 181
    scope.check()
    clock[0] = 600
    with pytest.raises(agent_module._Stop) as error:
        scope.check()
    assert error.value.reason == "timeout"


def test_cli_accepts_qwen_profile_model_flags_and_passes_frozen_profile(monkeypatch, tmp_path):
    profile = Q38.as_dict()
    profile["cache_namespace"] = "q38_album"
    captured = []
    monkeypatch.setattr(runtime_clients, "build_clients", lambda *args, **kwargs: (ProfileClient(), object(), object(), profile))
    monkeypatch.setattr(pipeline, "build", lambda *args, **kwargs: captured.append(kwargs) or {"status": "ok"})
    assert cli.main(["build", "synthetic.json", "-o", str(tmp_path), "--experiment", "Q38",
                     "--vlm-model", "qwen3.8-flash", "--embedding-model", "qwen"]) == 0
    assert captured[0]["experiment_config"] == profile
    assert captured[0]["concurrency"] == 4


@pytest.mark.parametrize("options", [["--vlm-model", "qwen3.7-flash"], ["--embedding-model", "wemm"],
                                     ["--concurrency", "2"]])
def test_cli_rejects_ignored_or_conflicting_flags_before_build(monkeypatch, tmp_path, options):
    monkeypatch.setattr(runtime_clients, "build_clients", lambda *args, **kwargs: (ProfileClient(), object(), object(), Q38.as_dict()))

    def forbidden(*args, **kwargs):
        raise AssertionError("Build must not start on conflicting model flags")

    monkeypatch.setattr(pipeline, "build", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["build", "synthetic.json", "-o", str(tmp_path), "--experiment", "Q38", *options])
    assert error.value.code == 2


def test_qwen_query_length_error_reports_2048(monkeypatch):
    import json

    class TooLong(RuntimeError):
        code = "query_too_long"

    class QueryClient(ProfileClient):
        def chat(self, messages, **kwargs):
            return {"role": "assistant", "content": None, "tool_calls": [{"id": "q", "type": "function",
                "function": {"name": "search", "arguments": json.dumps({"query": "long"})}}]}

    value = Agent("unused", llm=QueryClient(), library=profile_library(), max_steps=1)
    monkeypatch.setattr(value, "_execute", lambda *args, **kwargs: (_ for _ in ()).throw(TooLong()))
    answer = value.ask("question")
    assert answer.tool_trace[0]["result"]["error"]["max_tokens"] == 2048
