"""
Version-13: the Evaluation tab's data (the latest run, else the sample run) and issues shown
only where they belong: eval failures under Evaluation, users' failures on the Dashboard.
"""
import json

import pytest
from fastapi.testclient import TestClient

import backend.eval_api as eval_api
from backend import usage


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    s = usage.UsageStore(tmp_path / "usage.sqlite")
    monkeypatch.setattr(usage, "usage_store", s)
    return s


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app)


def test_the_sample_run_is_shown_until_a_real_one_exists(client, tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "RUNS_DIR", tmp_path / "runs")
    s = client.get("/api/eval/summary").json()
    assert s["sample"] is True and s["total"] == 40 and s["correct"] == 28 and s["accuracy"] == 70.0
    assert sum(d["total"] for d in s["by_difficulty"].values()) == 40
    assert sum(s["by_outcome"].values()) == 40

    (tmp_path / "runs").mkdir()
    run = {"benchmark": "BIRD dev", "results": [
        {"id": "1", "db_id": "a", "difficulty": "simple", "question": "q1", "correct": True, "outcome": "right", "llm_calls": 5, "cost": 0.002},
        {"id": "2", "db_id": "a", "difficulty": "challenging", "question": "q2", "correct": False, "outcome": "SQL error", "llm_calls": 8, "cost": 0.003},
    ]}
    (tmp_path / "runs" / "2026-11-04.json").write_text(json.dumps(run))
    s = client.get("/api/eval/summary").json()
    assert s["sample"] is False and (s["total"], s["correct"], s["accuracy"]) == (2, 1, 50.0)
    assert s["by_difficulty"]["challenging"]["accuracy"] == 0.0 and s["by_difficulty"]["moderate"]["accuracy"] is None
    assert (s["llm_calls"], s["avg_llm_calls"]) == (13, 6.5)


def test_results_filter_page_and_link_to_their_issue(client, store, tmp_path, monkeypatch):
    monkeypatch.setattr(eval_api, "RUNS_DIR", tmp_path / "runs")
    wrong = client.get("/api/eval/results", params={"filter": "wrong"}).json()
    assert wrong["total"] == 12 and all(not r["correct"] for r in wrong["items"]) and len(wrong["items"]) == 10
    assert client.get("/api/eval/results", params={"filter": "wrong", "page": 2}).json()["items"][-1]["id"]
    assert client.get("/api/eval/results", params={"difficulty": "challenging"}).json()["total"] == 6
    assert client.get("/api/eval/results", params={"search": "superhero"}).json()["total"] == 4

    question = wrong["items"][0]["question"]
    issue_id = store.add_issue("eval", question, {"sql": "x"})
    item = client.get("/api/eval/results", params={"filter": "wrong"}).json()["items"][0]
    assert item["issue"] == {"id": issue_id, "status": "new"}


def test_eval_issues_and_user_issues_are_shown_apart(client, store):
    store.add_issue("eval", "eval question", {})
    store.add_issue("user_report", "user question", {})
    store.add_issue("sql_error", "broken question", {})
    dashboard = client.get("/api/issues", params={"scope": "user_report,sql_error"}).json()
    assert dashboard["total"] == 2 and dashboard["by_source"] == {"user_report": 1, "sql_error": 1}
    assert dashboard["by_status"] == {"new": 2}
    evaluation = client.get("/api/issues", params={"scope": "eval"}).json()
    assert [i["question"] for i in evaluation["items"]] == ["eval question"]
    assert store.new_issue_ids(5, ["eval"]) == [evaluation["items"][0]["id"]]
    assert client.get("/api/issues").json()["total"] == 3  # no scope: everything, as before


def test_a_doubtful_bird_answer_is_labelled_but_still_counts_as_wrong(tmp_path, monkeypatch):
    """Version-19: our hand-checked list labels a BIRD answer as doubtful; the score never changes."""
    import json

    import backend.eval_api as eval_api

    doubtful = tmp_path / "bird_doubtful.json"
    doubtful.write_text(json.dumps({"questions": {"1481": {"reason": "uses the whole table's minimum"}}}))
    monkeypatch.setattr(eval_api, "DOUBTFUL_FILE", doubtful)
    run = {"suite": "bird_minidev", "results": [
        {"id": 1471, "question": "a", "correct": True},
        {"id": 1481, "question": "b", "correct": False},
        {"id": 1484, "question": "c", "correct": False},
    ]}
    summary = eval_api.summarize(run)
    assert (summary["correct"], summary["accuracy"], summary["doubtful"]) == (1, 33.3, 1)  # still wrong
    labelled = {r["id"]: r.get("doubtful") for r in eval_api.label_doubtful(run)}
    assert labelled == {1471: None, 1481: "uses the whole table's minimum", 1484: None}

    # the practice set is never labelled, even with the same id
    assert not eval_api.summarize({**run, "suite": "practice"})["doubtful"]
