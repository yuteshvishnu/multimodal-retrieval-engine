"""Scout Version-10: the steps the user checks, and 👍 / 👎 feedback."""
import json
import types

import pytest

import backend.usage as usage
from backend.text_to_sql.planner import StepPlanner


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


def test_planner_parses_steps_and_strips_numbers():
    assert StepPlanner._parse('{"steps": ["1. Take all orders", "2) Count them", ""]}') == ["Take all orders", "Count them"]
    with pytest.raises(ValueError):
        StepPlanner._parse('{"steps": []}')


def test_rerun_sends_the_comments_on_the_wrong_steps():
    prompt = StepPlanner.prompt(
        "orders per store", "count orders per store", "schema", "",
        [{"steps": ["Take orders that are not cancelled", "Count per store"],
          "comments": [{"step": 1, "comment": "count cancelled orders too"}]}],
    )
    assert "1. Take orders that are not cancelled" in prompt
    assert "step 1: count cancelled orders too" in prompt


@pytest.fixture(scope="module")
def pipeline():
    from backend.text_to_sql.pipeline import TextToSQLPipeline

    return TextToSQLPipeline()


def test_plan_then_sql_follows_the_confirmed_steps(pipeline, temp_usage):
    seen = []
    pipeline.planner.api_key = "x"
    pipeline.planner.client = fake_client(['{"steps": ["Take all stores", "Count them"]}'], seen)
    qid = temp_usage.start_question("scout", "How many stores?")
    plan = pipeline.plan("How many stores?", "Count the stores", question_id=qid)
    assert plan["steps"] == ["Take all stores", "Count them"] and plan["round"] == 1

    pipeline.generator.api_key = "x"
    pipeline.generator.client = fake_client(["SELECT COUNT(*) FROM stores"] * 3, seen)
    result = pipeline.run("How many stores?", "Count the stores", question_id=qid, steps=plan["steps"])
    assert result["rows"] == [[50]] and result["steps"] == plan["steps"]
    assert "Steps confirmed by the user" in seen[-1] and "2. Count them" in seen[-1]

    q = temp_usage.questions()["items"][0]
    assert {"step": "Steps, try 1", "calls": 1} in q["calls_per_try"]


def test_steps_stop_after_the_last_try(pipeline):
    tries = [{"steps": ["x"], "comments": [{"step": 1, "comment": "no"}]}] * 5
    assert "after 5 tries" in pipeline.plan("How many stores?", "Count the stores", history=tries)["error"]


def test_feedback_is_counted_and_a_thumbs_down_is_an_issue(temp_usage):
    a = temp_usage.start_question("scout", "question a")
    b = temp_usage.start_question("scout", "question b")
    temp_usage.add_feedback(a, "up")
    temp_usage.add_feedback(b, "up")
    temp_usage.add_feedback(b, "down", "cancelled orders are missing", {"sql": "SELECT 1", "steps": ["s1"]})  # latest counts

    s = temp_usage.summary()
    assert (s["thumbs_up"], s["thumbs_down"]) == (1, 1)
    issues = temp_usage.questions(feedback="down")["items"]
    assert [q["id"] for q in issues] == [b]
    assert issues[0]["feedback"]["comment"] == "cancelled orders are missing"
    assert issues[0]["feedback"]["details"]["sql"] == "SELECT 1"
    with pytest.raises(KeyError):
        temp_usage.add_feedback("no-such-question", "up")


def test_feedback_api_needs_a_comment_for_thumbs_down(temp_usage, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.text_to_sql.api as sql_api
    from backend.main import app

    monkeypatch.setattr(sql_api, "usage_store", temp_usage)
    qid = temp_usage.start_question("scout", "q")
    c = TestClient(app)
    assert c.post("/api/sql/feedback", data={"question_id": qid, "rating": "down"}).status_code == 400
    ok = c.post("/api/sql/feedback", data={"question_id": qid, "rating": "down", "comment": "wrong", "details": json.dumps({"sql": "x"})})
    assert ok.status_code == 200 and temp_usage.summary()["thumbs_down"] == 1


def test_database_names_become_plain_words():
    from backend.text_to_sql.wording import plain_words

    names = ["orders", "order_items", "order_date", "employee_id", "unit_price_cents", "status", "city"]
    assert plain_words("Count orders by order_date and employee_id", names) == "Count orders by order date and employee ID"
    assert plain_words("Use `order_items` and orders.status", names) == "Use order items and status"
    assert plain_words("Sum unit_price_cents per city", names) == "Sum unit price cents per city"
    # words that aren't database names, and plain single words, stay as they are
    assert plain_words("A follow_up question about status", names) == "A follow_up question about status"
    # brackets stay as written, with no tags
    assert plain_words("the orders (order_items) in March", names) == "the orders (order items) in March"


def test_tables_used_come_from_the_final_sql():
    from backend.text_to_sql.checks import tables_in_sql

    known = ["orders", "customers", "stores", "order_items"]
    assert tables_in_sql("SELECT COUNT(*) FROM orders WHERE status = 'cancelled'", known) == ["orders"]
    # WITH step names aren't tables; subqueries count
    sql = ("WITH c AS (SELECT store_id, COUNT(*) n FROM orders GROUP BY store_id) "
           "SELECT * FROM c JOIN stores s ON s.store_id = c.store_id WHERE s.store_id IN (SELECT store_id FROM customers)")
    assert sorted(tables_in_sql(sql, known)) == ["customers", "orders", "stores"]


def test_a_value_the_data_lacks_becomes_a_note_for_the_meaning():
    from backend.text_to_sql.pipeline import TextToSQLPipeline

    problem = "city has no value 'New York City'; its values are 'NY', 'NYC'."
    note = TextToSQLPipeline._missing_values([{"problems": [problem]}, {"problems": [problem, "other"]}])
    assert note == f"Scout checked the data: {problem} Use the values the data actually stores."
    assert TextToSQLPipeline._missing_values([{"problems": ["Dividing cents by 100"]}]) == ""
