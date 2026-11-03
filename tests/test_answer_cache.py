"""
Scout Version-12: a recent answer is reused when the same question comes back in other words.
The matching uses the real embedding model (no LLM); a stand-in LLM writes meanings and SQL.
"""
import json
import types

import pytest

import backend.usage as usage
from backend.rag.encoders.text_encoder import get_text_encoder
from backend.text_to_sql.answer_cache import AnswerCache


def fake_client(replies, seen):
    """A stand-in LLM: returns the next reply, records each prompt it was sent."""
    def create(**kwargs):
        seen.append(kwargs["messages"][-1]["content"])
        u = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20, completion_tokens_details=None)
        return types.SimpleNamespace(usage=u, choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=replies.pop(0)))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


@pytest.fixture(autouse=True)
def temp_usage(tmp_path, monkeypatch):
    store = usage.UsageStore(tmp_path / "usage.sqlite")
    monkeypatch.setattr(usage, "usage_store", store)
    return store


@pytest.fixture
def cache():
    return AnswerCache(10, get_text_encoder().encode)


def test_the_same_question_in_other_words_matches(cache):
    cache.add("How many orders were placed in March 2026?", "Count all orders placed in March 2026", [], "SELECT 1")
    cache.add("For each store, which employee handled the most orders?", "Per store, the top employee", [], "SELECT 2")
    for question in ("Number of orders placed during March 2026", "count the orders from march 2026",
                     "how many orders did we get in March this year"):
        assert cache.match(question)["sql"] == "SELECT 1", question
    assert cache.match("per store, who processed the most orders")["sql"] == "SELECT 2"


def test_close_but_different_questions_do_not_match(cache):
    cache.add("How many orders were placed in March 2026?", "Count all orders placed in March 2026", [], "SELECT 1")
    cache.add("For each store, which employee handled the most orders?", "Per store, the top employee", [], "SELECT 2")
    assert cache.match("How many orders were placed in April 2026?") is None  # embeddings 0.99 alike
    assert cache.match("How many orders were placed in March 2025?") is None
    assert cache.match("How many customers signed up in March 2026?") is None
    assert cache.match("Which employee handled the most orders overall?") is None  # per store vs overall


def test_what_is_saved_and_for_how_long(cache):
    assert cache.add("Orders placed last week", "Count orders from the last 7 days", [], "SELECT 1") is None
    assert cache.add("How many orders today?", "Count today's orders", [], "SELECT 1") is None
    for i in range(12):
        cache.add(f"question {i}", f"meaning {i}", [], "SELECT 1")
    assert len(cache) == 10  # the oldest two were dropped
    cache.add("question again", "meaning 11", [], "SELECT 2")  # the same meaning replaces the older copy
    assert len(cache) == 10


def test_an_option_that_only_rewords_the_saved_meaning_is_dropped(cache):
    # seen in a real run: the LLM was told not to repeat the saved meaning, and did
    saved = "Count every order whose order date falls in March 2026, no matter what its status is.\n- Include cancelled orders: yes"
    assert cache.repeats(saved, "Count every order whose order date falls in March 2026, regardless of its status.\n- Cancelled orders: included")
    # 0.94 alike as embeddings, but a different meaning: it narrows what's counted
    assert not cache.repeats(saved, "Count only the orders placed in March 2026 that are not cancelled.\n- Exclude cancelled orders: yes")
    assert not cache.repeats(saved, "Count every order placed in April 2026, whatever its status.\n- Cancelled orders: included")


def test_a_reported_answer_is_forgotten(cache):
    entry_id = cache.add("How many stores?", "Count the stores", [], "SELECT COUNT(*) FROM stores", "q1")
    cache.link(entry_id, "q2")  # a later question answered from it
    assert cache.forget_question("q2") and cache.get(entry_id) is None


@pytest.fixture(scope="module")
def pipeline():
    from backend.text_to_sql.pipeline import TextToSQLPipeline

    return TextToSQLPipeline()


def test_a_saved_answer_is_the_first_option_and_answers_without_llm_calls(pipeline, temp_usage):
    pipeline.answers.clear()
    seen = []
    pipeline.generator.api_key = "x"
    pipeline.generator.client = fake_client(["SELECT COUNT(*) FROM stores"] * 3, seen)
    first = pipeline.run("How many stores are there?", "Count all the stores\n- Closed stores: counted",
                         question_id="q1", steps=["Take all stores", "Count them"])
    assert first["rows"] == [[50]] and len(pipeline.answers) == 1

    # asked again in other words: the saved meaning comes first, the LLM writes the others
    pipeline.interpreter.api_key = "x"
    pipeline.interpreter.client = fake_client(
        ['{"options": [{"meaning": "Count all the stores", "choices": ["Closed stores: counted"]}, '
         '{"meaning": "Count only the stores that are not closed", "choices": []}]}'], seen)
    options = pipeline.interpret("What is the number of stores?")["options"]
    assert options[0]["meaning"] == "Count all the stores" and options[0]["choices"] == ["Closed stores: counted"]
    assert options[0]["reuse_id"] and [o["meaning"] for o in options[1:]] == ["Count only the stores that are not closed"]
    assert "already offered to the user as the first option" in seen[-1]

    answer = pipeline.reuse(options[0]["reuse_id"], "q2")
    assert answer["rows"] == [[50]] and answer["llm_calls"] == 0 and answer["reused"]
    assert answer["steps"] == ["Take all stores", "Count them"]

    assert pipeline.reuse("gone", "q3") == {"fallback": True, "llm_calls": 0}
    pipeline.reload()  # a table changed: nothing saved can be trusted
    assert len(pipeline.answers) == 0


def test_follow_ups_are_not_saved(pipeline):
    pipeline.answers.clear()
    pipeline.generator.api_key = "x"
    pipeline.generator.client = fake_client(["SELECT COUNT(*) FROM stores"] * 3, [])
    pipeline.run("now count them", "Count the stores", previous={"question": "List stores", "sql": "SELECT * FROM stores"})
    assert len(pipeline.answers) == 0


def test_reporting_an_answer_stops_it_being_reused(temp_usage, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.text_to_sql.api as sql_api
    from backend.main import app

    monkeypatch.setattr(sql_api, "usage_store", temp_usage)
    qid = temp_usage.start_question("scout", "How many stores?")
    sql_api.pipeline.answers.add("How many stores?", "Count the stores", [], "SELECT COUNT(*) FROM stores", qid)
    TestClient(app).post("/api/sql/feedback", data={"question_id": qid, "rating": "down", "comment": "wrong",
                                                    "details": json.dumps({})})
    assert sql_api.pipeline.answers.match("How many stores?") is None
