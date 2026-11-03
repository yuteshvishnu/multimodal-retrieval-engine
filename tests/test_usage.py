"""Dashboard: every LLM call recorded per question, with tokens, time and cost."""
import types
from datetime import date, datetime

import pytest

import backend.usage as usage
from backend.config import LLM_PRICES
from backend.llm import chat


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = usage.UsageStore(tmp_path / "usage.sqlite")
    monkeypatch.setattr(usage, "usage_store", s)  # chat() records into this one
    return s


def api_usage(prompt, completion, reasoning=0):
    """What the API returns in `usage`."""
    return types.SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning),
    )


def test_cost_is_tokens_times_price():
    price_in, price_out = LLM_PRICES["openai/gpt-oss-120b"]
    _, _, cost = usage.call_cost("openai/gpt-oss-120b", 1_000_000, 500_000)
    assert cost == pytest.approx(price_in + price_out / 2)
    assert usage.call_cost("some/unpriced-model", 1000, 1000)[2] == 0


def test_question_metrics(store):
    qid = store.start_question("scout", "How many orders in March?")
    assert store.start_question("scout", "same question", qid) == qid      # later steps reuse the id
    model = "openai/gpt-oss-120b"
    store.record_call({"question_id": qid, "stage": "meaning", "round": 1}, model, api_usage(1000, 200, 150), 0.5, "ok")
    store.record_call({"question_id": qid, "stage": "meaning", "round": 2}, model, api_usage(1000, 200), 0.5, "ok")
    for round_no in (1, 1, 1, 2, 2, 2):
        store.record_call({"question_id": qid, "stage": "sql", "round": round_no}, model, api_usage(2000, 300), 1.0, "ok")
    store.add_time(qid, 3.0)
    store.finish(qid, "answered", "medium")
    store.record_call({"stage": "table_description"}, model, api_usage(500, 100), 0.2, "ok")  # not a question

    q = store.questions()["items"][0]
    assert q["llm_calls"] == 8
    assert q["calls_per_try"] == [
        {"step": "Meaning, try 1", "calls": 1}, {"step": "Meaning, try 2", "calls": 1},
        {"step": "SQL, round 1", "calls": 3}, {"step": "SQL, round 2", "calls": 3},
    ]
    assert q["retries"] == 1 + 3                                             # one re-ask + round 2
    assert (q["input_tokens"], q["output_tokens"], q["reasoning_tokens"]) == (14_000, 2_200, 150)
    assert q["llm_latency_ms"] == pytest.approx(7000) and q["total_latency_ms"] == pytest.approx(3000)
    assert (q["outcome"], q["confidence"]) == ("answered", "medium")
    assert q["cost"] == pytest.approx(usage.call_cost(model, 14_000, 2_200)[2])

    s = store.summary()
    assert (s["questions"], s["llm_calls"], s["other_calls"]) == (1, 9, 1)
    assert store.questions(search="march")["total"] == 1 and store.questions(search="nothing")["total"] == 0
    day = store.costs("day", datetime.now().date())
    assert len(day["buckets"]) == 24 and day["total_calls"] == 9
    assert day["total_cost"] == pytest.approx(s["cost"])
    assert any(q["id"] == qid for b in day["buckets"] for q in b["questions"])


def test_failed_call_is_recorded_and_still_raises(store):
    class Broken:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise TimeoutError("no answer")

    with pytest.raises(TimeoutError):
        chat(Broken, {"question_id": "q1", "stage": "sql", "round": 1}, model="openai/gpt-oss-120b", messages=[])
    row = store._read("SELECT status, input_tokens, cost FROM llm_calls")[0]
    assert tuple(row) == ("error", 0, 0)


def test_paging_is_10_per_page(store):
    for i in range(23):
        store.start_question("scribe", f"question {i}")
    first, last = store.questions(page=1), store.questions(page=3)
    assert (first["pages"], len(first["items"]), len(last["items"])) == (3, 10, 3)


def test_period_slots_are_fixed(store):
    assert len(store.costs("week", date(2026, 11, 3))["buckets"]) == 7
    assert len(store.costs("month", date(2026, 2, 10))["buckets"]) == 28
    assert len(store.costs("year", date(2026, 6, 1))["buckets"]) == 12
    assert store.costs("week", date(2026, 11, 5))["buckets"][0]["label"] == "Mon 2"   # weeks start on Monday
    with pytest.raises(ValueError):
        store.costs("decade", date(2026, 1, 1))


def test_an_empty_question_is_not_recorded(store):
    assert store.start_question("scout", "   ") is None
    assert store.summary()["questions"] == 0
