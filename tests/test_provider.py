"""Version-16: the LLM provider. On OpenRouter every call is pinned to our hosts and priced exactly."""
import types

import backend.llm as llm
from backend import usage


def test_openrouter_calls_are_pinned_and_priced_by_what_they_cost(tmp_path, monkeypatch):
    store = usage.UsageStore(tmp_path / "usage.sqlite")
    monkeypatch.setattr(usage, "usage_store", store)
    monkeypatch.setattr(llm, "LLM_PROVIDER", "openrouter")
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        u = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=100, completion_tokens_details=None, cost=0.00004)
        return types.SimpleNamespace(usage=u)

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    llm.chat(client, {"question_id": "q", "stage": "sql"}, model="openai/gpt-oss-120b", messages=[])

    provider = sent["extra_body"]["provider"]
    assert provider == {"order": llm.OPENROUTER_PROVIDERS, "allow_fallbacks": False}
    assert sent["extra_body"]["usage"] == {"include": True}
    assert store.calls_for("q") == {"llm_calls": 1, "tokens": 1100, "cost": 0.00004}  # the reported cost, not our estimate


def test_groq_calls_are_sent_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(usage, "usage_store", usage.UsageStore(tmp_path / "usage.sqlite"))
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return types.SimpleNamespace(usage=None)

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    llm.chat(client, None, model="openai/gpt-oss-120b", messages=[])
    assert "extra_body" not in sent
