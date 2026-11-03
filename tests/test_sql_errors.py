"""Scout Version-9: the error book (known SQLite errors, hints, early stop, logging)."""
import json

import pytest

from backend.text_to_sql.sql_errors import KNOWN_FILE, ErrorBook, normalize

# a real query that triggers each known error on SQLite
BAD_SQL = {
    "distinct-aggregate-separator": "SELECT GROUP_CONCAT(DISTINCT city, ', ') FROM customers",
    "date-functions": "SELECT YEAR(signup_date) FROM customers",
    "extract": "SELECT EXTRACT(YEAR FROM signup_date) FROM customers",
    "interval": "SELECT * FROM orders WHERE order_date > CURRENT_DATE - INTERVAL '7 days'",
    "ilike": "SELECT * FROM customers WHERE city ILIKE 'new%'",
    "top-or-fetch": "SELECT TOP 5 * FROM customers",
    "aggregate-in-where": "SELECT store_id FROM orders WHERE COUNT(*) > 2 GROUP BY store_id",
    "window-in-where": "SELECT * FROM orders WHERE ROW_NUMBER() OVER (ORDER BY order_id) = 1",
    "having-without-group-by": "SELECT store_id FROM orders HAVING COUNT(*) > 2",
    "unknown-column": "SELECT statuss FROM orders",
    "unknown-table": "SELECT * FROM sales",
    "reserved-word": "SELECT * FROM order",
    "ambiguous-column": "SELECT store_id FROM orders o JOIN stores s ON s.store_id = o.store_id",
    "subselect-columns": "SELECT * FROM orders WHERE customer_id IN (SELECT customer_id, city FROM customers)",
    "wrong-argument-count": "SELECT ROUND() FROM orders",
}


@pytest.fixture
def book(tmp_path):
    return ErrorBook(log_file=tmp_path / "log.jsonl")


def sqlite_error(database, sql):
    try:
        database.explain(sql)
    except Exception as e:
        return f"Invalid SQL: {e}"
    raise AssertionError(f"expected an error: {sql}")


def test_every_entry_is_complete_and_tested():
    entries = json.loads(KNOWN_FILE.read_text(encoding="utf-8"))
    for entry in entries:
        assert {"id", "match", "hint", "rule", "example_bad", "example_fix"} <= entry.keys(), entry["id"]
    assert {e["id"] for e in entries} == set(BAD_SQL), "add a BAD_SQL case for every entry"


@pytest.mark.parametrize("known_id, sql", BAD_SQL.items())
def test_real_sqlite_errors_match_their_entry(database, book, known_id, sql):
    entry = book.match(sqlite_error(database, sql), sql)
    assert entry is not None and entry["id"] == known_id


def test_unrelated_error_stays_unknown(database, book):
    sql = "SELECT * FROM orders WHERE (status = 'x'"
    assert book.match(sqlite_error(database, sql), sql) is None


def test_normalize_groups_the_same_kind_of_error():
    assert normalize("Invalid SQL: no such column: o.statuss") == "no such column: <name>"
    assert normalize('near "5": syntax error') == normalize('near "10": syntax error')
    assert normalize("sub-select returns 2 columns - expected 1") == "sub-select returns <n> columns - expected <n>"


def test_rules_put_the_most_seen_errors_first(book):
    book.log({"type": "error", "known_id": "ilike"})
    assert book.rules()[0] == book.match('near "ILIKE": syntax error')["rule"]
    assert len(book.rules()) <= 10


# ---------- in the pipeline, with a stand-in LLM ----------

@pytest.fixture(scope="module")
def pipeline():
    from backend.text_to_sql.pipeline import TextToSQLPipeline

    return TextToSQLPipeline()


def fake_llm(pipeline, first_sql, later_sql, seen_feedback):
    """Writes first_sql until it gets feedback, then later_sql; records the feedback."""
    def generate(question, schema, budget, context="", feedback="", **kwargs):
        budget.take()
        seen_feedback.append(feedback)
        return later_sql if feedback else first_sql
    pipeline.generator.generate = generate


def test_known_error_gets_its_hint_and_is_fixed(pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "errors", ErrorBook(log_file=tmp_path / "log.jsonl"))
    feedback = []
    fake_llm(pipeline, BAD_SQL["distinct-aggregate-separator"], "SELECT GROUP_CONCAT(DISTINCT city) FROM customers", feedback)

    result = pipeline.run("cities of customers", "every city once")

    assert not result.get("error") and result["rows"]
    assert "GROUP_CONCAT(DISTINCT city)" in feedback[-1]           # round 2 was told the fix
    log = pipeline.errors.read_log()
    assert [e["known_id"] for e in log if e["type"] == "error"] == ["distinct-aggregate-separator"] * 3
    assert [e["fixed"] for e in log if e["type"] == "outcome"] == [True]


def test_unknown_error_everywhere_stops_after_round_1(pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "errors", ErrorBook(log_file=tmp_path / "log.jsonl"))
    feedback = []
    unknown = "SELECT * FROM orders WHERE (status = 'x'"
    fake_llm(pipeline, unknown, unknown, feedback)

    result = pipeline.run("orders", "all orders")

    assert result["llm_calls"] == 3                                 # round 2 skipped
    assert result["error"].startswith("Scout couldn't build a working query")
    assert "first round" in result["error"] and "Invalid SQL" not in result["error"]
    outcome = [e for e in pipeline.errors.read_log() if e["type"] == "outcome"][0]
    assert outcome["stopped_early"] and not outcome["fixed"]


def test_rules_reach_the_sql_prompt(pipeline):
    prompt = pipeline.generator._system_prompt()
    assert "SQLite rules" in prompt and "GROUP_CONCAT(DISTINCT x) takes no separator" in prompt
