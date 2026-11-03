"""
Version-11: the Failure Analyst agent and its issues. A stand-in LLM plays the agent (tool
calls, then a report), so no API key is needed; the tools run on the practice database.
"""
import json
import shutil
import types

import pytest

from backend import usage
from backend.agent.analyst import FailureAnalyst
from backend.agent.fixes import apply_fix, entry_checks
from backend.agent.tools import ReadOnlyTools
from backend.text_to_sql.sql_errors import KNOWN_FILE, ErrorBook


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    s = usage.UsageStore(tmp_path / "usage.sqlite")
    monkeypatch.setattr(usage, "usage_store", s)
    return s


@pytest.fixture
def pipeline(database, row_counts, tmp_path):
    """Just what the agent uses, with a copy of the error book so tests can't change the real one."""
    known = tmp_path / "sql_errors.json"
    shutil.copy(KNOWN_FILE, known)
    return types.SimpleNamespace(
        tables=database.tables, database=database, row_counts=row_counts,
        errors=ErrorBook(known_file=known, log_file=tmp_path / "log.jsonl"),
    )


def tool_call(name, args, i=0):
    return types.SimpleNamespace(id=f"call_{i}", function=types.SimpleNamespace(name=name, arguments=json.dumps(args)))


def agent_llm(turns, seen):
    """A stand-in agent: each turn is a list of (tool, arguments); records what it was offered."""
    def create(**kwargs):
        seen.append(kwargs)
        calls = [tool_call(name, args, i) for i, (name, args) in enumerate(turns.pop(0))]
        message = types.SimpleNamespace(content="", reasoning="Let me check the data.", tool_calls=calls)
        u = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=100, completion_tokens_details=None)
        return types.SimpleNamespace(usage=u, choices=[types.SimpleNamespace(message=message)])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


REPORT = {
    "what_happened": "Scout counted 0 stores.", "why": "It filtered on a city that isn't spelled that way.",
    "root_cause": "data", "fix_type": "none", "fix": "Nothing to change.", "confidence": "high",
    "fixed_sql": "SELECT COUNT(*) FROM stores",
}


def test_the_same_failure_updates_one_issue(store):
    a = store.add_issue("eval", "How many stores?", {"sql": "x"})
    store.set_issue(a, "rejected", note="not now")
    b = store.add_issue("eval", "how many  STORES?", {"sql": "y"})  # same question, same source
    assert a == b
    issue = store.issue(a)
    assert issue["status"] == "new" and issue["details"]["sql"] == "y" and issue["note"] is None
    store.add_issue("sql_error", "How many stores?", {})            # another source: its own issue
    assert store.issues()["by_source"] == {"eval": 1, "sql_error": 1}
    with pytest.raises(ValueError):
        store.add_issue("rumour", "q", {})


def test_an_issue_is_investigated_once_at_a_time(store):
    i = store.add_issue("eval", "q", {})
    assert store.start_investigation(i) is True
    assert store.start_investigation(i) is False
    assert store.new_issue_ids(5) == []
    assert usage.UsageStore(store.path).issue(i)["status"] == "new"  # a restart frees it


def test_tools_only_read(pipeline):
    tools = ReadOnlyTools(pipeline)
    assert tools.call("run_query", {"sql": "DELETE FROM stores"}).startswith("Refused")
    assert tools.call("run_query", {"sql": "SELECT 1; DROP TABLE stores"}).startswith("Refused")
    assert json.loads(tools.call("run_query", {"sql": "SELECT COUNT(*) FROM stores"}))["rows"] == [[50]]
    assert "SQLite error" in tools.call("run_query", {"sql": "SELECT nope FROM stores"})
    assert "No table" in tools.call("describe_table", {"table": "nope"})
    assert "No column" in tools.call("column_values", {"table": "stores", "column": "nope"})
    assert json.loads(tools.call("column_values", {"table": "stores", "column": "city"}))["distinct_values"] > 0
    assert tools.call("write_file", {}).startswith("Unknown tool")


def test_agent_investigates_then_reports_and_code_tests_the_fix(store, pipeline):
    issue_id = store.add_issue("eval", "How many stores?", {"sql": "SELECT 0", "rows": [[0]], "gold_sql": "SELECT COUNT(*) FROM stores"})
    store.start_investigation(issue_id)
    seen = []
    llm = agent_llm([[("column_values", {"table": "stores", "column": "city"})], [("submit_report", REPORT)]], seen)
    FailureAnalyst(pipeline, store, client=llm).investigate(issue_id)

    issue = store.issue(issue_id)
    assert issue["status"] == "report_ready" and issue["report"]["why"] == REPORT["why"]
    assert issue["report"]["test"]["passed"] is True  # its SQL gives the correct answer's rows
    assert [s["kind"] for s in issue["steps"]] == ["start", "reasoning", "tool", "reasoning", "report", "test"]
    assert issue["steps"][2]["title"] == "Checked the values in stores.city"
    assert issue["agent_calls"] == 2 and issue["agent_cost"] > 0
    assert "Correct SQL: SELECT COUNT(*) FROM stores" in seen[0]["messages"][1]["content"]
    assert store.summary()["llm_calls"] == 2  # priced like every other call


def test_agent_must_report_when_its_tool_calls_run_out(store, pipeline):
    issue_id = store.add_issue("sql_error", "q", {})
    store.start_investigation(issue_id)
    seen = []
    turns = [[("list_tables", {})], [("list_tables", {})], [("submit_report", {**REPORT, "fixed_sql": ""})]]
    FailureAnalyst(pipeline, store, client=agent_llm(turns, seen), max_tool_calls=2).investigate(issue_id)
    assert [t["function"]["name"] for t in seen[2]["tools"]] == ["submit_report"]  # only the report is left
    assert store.issue(issue_id)["status"] == "report_ready"


def test_a_failed_fix_goes_back_to_the_agent_once(store, pipeline):
    issue_id = store.add_issue("eval", "How many stores?", {"gold_sql": "SELECT COUNT(*) FROM stores"})
    store.start_investigation(issue_id)
    seen = []
    wrong = {**REPORT, "fix_type": "prompt", "fixed_sql": "SELECT COUNT(*) - 1 FROM stores"}
    turns = [[("submit_report", wrong)], [("submit_report", {**REPORT, "fix_type": "prompt"})]]
    FailureAnalyst(pipeline, store, client=agent_llm(turns, seen)).investigate(issue_id)
    assert any("Our test of your fix failed" in m["content"] for m in seen[1]["messages"] if m["role"] == "tool")
    issue = store.issue(issue_id)
    assert issue["report"]["test"]["passed"] is True
    assert [s["title"] for s in issue["steps"] if s["kind"].startswith("test")] == [
        "Tested the fix: failed, sent back to the agent", "Tested the fix: passed"]


def test_a_broken_investigation_is_marked_failed(store, pipeline):
    issue_id = store.add_issue("eval", "q", {})
    store.start_investigation(issue_id)
    FailureAnalyst(pipeline, store, client=agent_llm([], [])).investigate(issue_id)  # the LLM gives nothing
    issue = store.issue(issue_id)
    assert issue["status"] == "failed" and issue["steps"][-1]["kind"] == "error"


def test_error_book_fix_is_checked_then_applied(pipeline):
    entry = {"id": "split-part", "match": r"no such function: SPLIT_PART", "hint": "Use substr and instr.",
             "rule": "no SPLIT_PART in SQLite: use substr / instr"}
    assert all(c["ok"] for c in entry_checks(entry, pipeline, ["no such function: SPLIT_PART"]))
    assert not all(c["ok"] for c in entry_checks(entry, pipeline, ["no such column: x"]))
    assert not entry_checks({**entry, "id": "date-functions"}, pipeline, [])[0]["ok"]  # already in the book

    issue = {"id": "abc", "report": {"fix_type": "error_book", "error_book_entry": entry}}
    assert "split-part" in apply_fix(issue, pipeline)
    assert pipeline.errors.match("no such function: SPLIT_PART")["id"] == "split-part"  # used from the next question
    assert apply_fix({"id": "x", "report": {"fix_type": "prompt"}}, pipeline).startswith("Approved")


def test_an_error_book_fix_for_a_query_that_ran_fails_its_test(pipeline):
    from backend.agent.fixes import test_fix

    issue = {"source": "eval", "details": {"gold_sql": "SELECT COUNT(*) FROM stores"}}
    report = {"fix_type": "error_book", "fixed_sql": "SELECT COUNT(*) FROM stores",
              "error_book_entry": {"id": "ties", "match": "ROW_NUMBER", "hint": "h", "rule": "r"}}
    test = test_fix(issue, report, pipeline)
    assert test["passed"] is False and "No SQLite error" in test["checks"][-1]["text"]


def test_a_report_needs_a_tested_corrected_query():
    from backend.agent.analyst import report_problems

    assert "fixed_sql" in report_problems({**REPORT, "fix_type": "prompt", "fixed_sql": ""})[0]
    assert report_problems({**REPORT, "fix_type": "none", "fixed_sql": ""}) == []
    assert report_problems(REPORT) == []


def test_a_user_report_becomes_an_issue(store, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.text_to_sql.api as sql_api
    from backend.main import app

    monkeypatch.setattr(sql_api, "usage_store", store)
    qid = store.start_question("scout", "How many stores?")
    c = TestClient(app)
    c.post("/api/sql/feedback", data={"question_id": qid, "rating": "down", "comment": "should be 50",
                                      "details": json.dumps({"sql": "SELECT 0"})})
    listed = c.get("/api/issues").json()
    assert listed["total"] == 1 and listed["agent"]["max_tool_calls"] > 0
    issue = listed["items"][0]
    assert (issue["source"], issue["question"], issue["details"]["comment"]) == ("user_report", "How many stores?", "should be 50")
    assert c.post(f"/api/issues/{issue['id']}/approve").status_code == 400  # no report yet
    assert c.post(f"/api/issues/{issue['id']}/reject", data={"note": "fine"}).json()["status"] == "rejected"
