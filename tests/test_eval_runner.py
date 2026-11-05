"""
Version-14: an evaluation run started from the Evaluation tab. A stand-in pipeline plays Scout
(no LLM): each question goes meaning -> steps -> SQL, is scored against the correct rows, saved
as it finishes, and a wrong one becomes an Eval issue.
"""
import json
import types

import pytest

from backend import usage
from backend.eval_runner import EvalRunner, estimate, load_questions, suites

QUESTIONS = load_questions("practice")[:3]
GOLD = {q["gold_sql"]: [[i]] for i, q in enumerate(QUESTIONS)}


def run_all(sql):
    """The stand-in database: a correct SQL gives its rows, Scout's "SELECT 7" gives 7."""
    return ["n"], GOLD[sql] if sql in GOLD else [[int(sql.split()[1])]]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    s = usage.UsageStore(tmp_path / "usage.sqlite")
    monkeypatch.setattr(usage, "usage_store", s)
    return s


def fake_pipeline(answers, store, seen, seen_history=None):
    """answers: one per call to run(), each a dict result; records one priced call per stage."""
    seen_history = [] if seen_history is None else seen_history
    u = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=200, completion_tokens_details=None)

    def call(question_id, stage):
        seen.append((question_id, stage))
        store.record_call({"question_id": question_id, "stage": stage}, "openai/gpt-oss-120b", u, 0.1, "ok")

    def interpret(question, history=(), question_id=None, use_saved=True):
        assert use_saved is False  # an eval never uses saved answers
        seen_history.append(list(history))
        call(question_id, "meaning")
        return {"options": [{"meaning": "Count them", "choices": ["Cancelled: counted"]}]}

    def plan(question, meaning, question_id=None):
        assert meaning == "Count them\n- Cancelled: counted"  # the first option, as the user would confirm it
        call(question_id, "steps")
        return {"steps": ["Take them", "Count them"]}

    def run(question, meaning, question_id=None, steps=None, save=True):
        assert save is False and steps == ["Take them", "Count them"]
        call(question_id, "sql")
        return answers.pop(0)

    return types.SimpleNamespace(
        interpret=interpret, plan=plan, run=run,
        database=types.SimpleNamespace(run_all=run_all),
        results=types.SimpleNamespace(get=lambda result_id: None),
    )


def test_a_run_answers_scores_saves_and_logs_wrong_answers(store, tmp_path):
    seen = []
    answers = [
        {"rows": [[0]], "sql": "SELECT 0", "confidence": "high"},                   # right
        {"rows": [[99]], "sql": "SELECT 99", "confidence": "medium"},                # wrong rows
        {"error": "Scout couldn't build a working query", "sql": "SELEC", "attempts": [{"status": "rejected", "error": "near \"SELEC\": syntax error"}]},
    ]
    runner = EvalRunner(fake_pipeline(answers, store, seen), tmp_path / "runs", sleep=lambda s: None)
    progress = runner.start("practice", last=3, wait=True)

    assert progress["status"] == "finished" and (progress["done"], progress["total"], progress["correct"]) == (3, 3, 1)
    assert progress["accuracy"] == 33.3 and progress["llm_calls"] == 9 and progress["cost"] > 0
    assert [r["outcome"] for r in progress["recent"]] == ["SQL error", "wrong rows", "right"]  # newest first
    assert [stage for _, stage in seen[:3]] == ["meaning", "steps", "sql"]
    assert all(tag.startswith(f"eval:{progress['id']}:") for tag, _ in seen)

    saved = json.loads((tmp_path / "runs" / f"{progress['id']}.json").read_text())
    assert saved["status"] == "finished" and saved["short"] == "Practice" and len(saved["results"]) == 3
    assert saved["results"][0]["difficulty"] == "simple" and saved["results"][0]["tokens"] == 3600
    issues = store.issues(scope=["eval"])
    assert issues["total"] == 2 and store.summary()["questions"] == 0  # not counted as users' questions


def test_rate_limits_wait_and_retry_then_stop(store, tmp_path):
    waits = []
    limited = {"error": "Could not understand the question: Error code: 429 rate limit"}
    answers = [limited, {"rows": [[0]], "sql": "SELECT 0"}, limited, limited, limited]
    runner = EvalRunner(fake_pipeline(answers, store, []), tmp_path / "runs", sleep=waits.append)
    progress = runner.start("practice", last=2, wait=True)

    assert progress["done"] == 1 and progress["correct"] == 1  # the first question got through on its 2nd try
    assert progress["status"] == "stopped" and "rate limit" in progress["message"]
    assert len(waits) == 4


def test_stop_then_resume_carries_on(store, tmp_path):
    answers = [{"rows": [[i]], "sql": f"SELECT {i}"} for i in range(3)]
    runner = EvalRunner(fake_pipeline(answers, store, []), tmp_path / "runs", sleep=lambda s: None)
    original_run = runner.pipeline.run

    def run_then_stop(*args, **kwargs):
        runner.stop()  # the user clicks Stop while question 1 is being answered
        return original_run(*args, **kwargs)

    runner.pipeline.run = run_then_stop
    first = runner.start("practice", last=3, wait=True)
    assert first["status"] == "stopped" and first["done"] == 1  # question 1 finished, nothing cut in half

    runner.pipeline.run = original_run
    resumed = runner.resume(first["id"], wait=True)
    assert resumed["status"] == "finished" and resumed["done"] == 3 and resumed["correct"] == 3
    with pytest.raises(ValueError):
        runner.resume(first["id"])  # a finished run can't be resumed


def test_suites_and_estimate(monkeypatch, tmp_path):
    import backend.eval_runner as eval_runner

    monkeypatch.setattr(eval_runner, "PROVIDER_TOKENS_PER_DAY", 200_000)  # Groq's free limit, whatever .env says
    monkeypatch.setitem(eval_runner.SUITES, "bird_minidev",
                        {**eval_runner.SUITES["bird_minidev"], "path": tmp_path / "missing.json"})  # not downloaded
    listed = {s["id"]: s for s in suites()}
    assert listed["practice"]["ready"] and listed["practice"]["questions"] == 20
    assert not listed["bird_minidev"]["ready"] and listed["bird_minidev"]["missing"]
    small, big = estimate(5), estimate(500)
    assert small["tokens"] == 40_000 and not small["over_daily_limit"]
    assert big["over_daily_limit"] and big["cost"] > small["cost"]


def test_the_scorecard_ignores_a_run_still_running(tmp_path, monkeypatch):
    import backend.eval_api as eval_api

    monkeypatch.setattr(eval_api, "RUNS_DIR", tmp_path)
    (tmp_path / "1.json").write_text(json.dumps({"short": "Practice", "status": "finished", "results": [
        {"id": 1, "question": "q", "difficulty": "simple", "correct": True, "outcome": "right"}]}))
    (tmp_path / "2.json").write_text(json.dumps({"status": "running", "results": []}))
    s = eval_api.summarize(eval_api.latest_run())
    assert (s["short"], s["accuracy"], s["sample"]) == ("Practice", 100.0, False)


def test_each_run_has_its_own_issues_and_is_listed_for_the_agent(store, tmp_path):
    """Version-15: the same wrong question in two runs is two issues, one per run."""
    wrong = {"rows": [[99]], "sql": "SELECT 99"}
    runner = EvalRunner(fake_pipeline([wrong, dict(wrong)], store, []), tmp_path / "runs", sleep=lambda s: None)
    first = runner.start("practice", last=1, wait=True)
    second = runner.start("practice", last=1, wait=True)

    for run in (first, second):
        listed = store.issues(scope=["eval"], run=run["id"])
        assert listed["total"] == 1 and listed["items"][0]["run_id"] == run["id"]
        assert store.new_issue_ids(5, ["eval"], run["id"]) == [listed["items"][0]["id"]]
    runs = runner.finished_runs()
    assert [r["id"] for r in runs] == [second["id"], first["id"]]  # newest first
    assert (runs[0]["wrong"], runs[0]["accuracy"]) == (1, 0.0)
    assert store.issue_counts_by_run()[first["id"]] == {"total": 1, "new": 1}


def test_issues_logged_before_run_ids_move_under_their_run(store, tmp_path):
    question = QUESTIONS[0]["question"]
    old = store.add_issue("eval", question, {})            # logged by a run before Version-15
    script = store.add_issue("eval", "a script question", {})  # logged by evaluate.py: no run
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "r1.json").write_text(json.dumps({
        "id": "r1", "status": "finished", "run_at": "2000-01-01T00:00:00", "planned": [1],
        "results": [{"id": 1, "question": question, "correct": False}],
    }))
    EvalRunner(None, runs).finished_runs()
    assert store.issue(old)["run_id"] == "r1" and store.issue(script)["run_id"] is None
    assert store.issue_counts_by_run() == {"r1": {"total": 1, "new": 1}, "none": {"total": 1, "new": 1}}


def test_a_value_not_in_the_data_asks_the_meaning_again_once(store, tmp_path):
    """The checks found the meaning asks for a value the data lacks: one more meaning, with the real values."""
    note = "Scout checked the data: city has no value 'Big Apple'; its values are 'NY'."
    missing = {"error": "Scout couldn't build a working query", "rethink_meaning": note,
               "attempts": [{"status": "rejected", "problems": [note]}]}
    histories = []
    runner = EvalRunner(fake_pipeline([dict(missing), {"rows": [[0]], "sql": "SELECT 0"}], store, [], histories),
                        tmp_path / "runs", sleep=lambda s: None)
    fixed = runner.start("practice", last=1, wait=True)
    assert fixed["correct"] == 1
    assert histories[0] == [] and histories[1][0]["comment"] == note  # the second meaning was told the real values

    runner = EvalRunner(fake_pipeline([dict(missing), dict(missing), dict(missing)], store, [], histories),
                        tmp_path / "runs", sleep=lambda s: None)
    stuck = runner.start("practice", last=1, wait=True)
    assert stuck["recent"][0]["outcome"] == "value not in data"  # stopped after 2 meanings, not 3


def test_outcomes_say_who_rejected_the_sql(store, tmp_path):
    sqlite_rejected = {"error": "x", "attempts": [{"status": "rejected", "error": "no such column: y"}]}
    checks_rejected = {"error": "x", "attempts": [{"status": "rejected", "problems": ["Dividing cents by 100"]}]}
    runner = EvalRunner(fake_pipeline([sqlite_rejected, checks_rejected], store, []), tmp_path / "runs",
                        sleep=lambda s: None)
    progress = runner.start("practice", last=2, wait=True)
    assert [r["outcome"] for r in progress["recent"]] == ["failed checks", "SQL error"]  # newest first



# ---------- Version-18: BIRD Mini-Dev, read from BIRD's own files ----------

def fake_bird(tmp_path, monkeypatch):
    """A 3-question, 2-database BIRD folder in BIRD's own layout and fields."""
    import sqlite3

    from backend import eval_runner

    root = tmp_path / "MINIDEV"
    for db, value in (("shop", 7), ("zoo", 9)):
        folder = root / "dev_databases" / db
        folder.mkdir(parents=True)
        con = sqlite3.connect(folder / f"{db}.sqlite")
        con.execute("CREATE TABLE t (x INTEGER)")
        con.execute("INSERT INTO t VALUES (?)", (value,))
        con.commit()
        con.close()
    questions = [
        {"question_id": 101, "db_id": "shop", "question": "How many?", "evidence": "x is the count", "SQL": "SELECT x FROM t", "difficulty": "simple"},
        {"question_id": 205, "db_id": "zoo", "question": "Which animal?", "evidence": "", "SQL": "SELECT x FROM t", "difficulty": "moderate"},
        {"question_id": 300, "db_id": "shop", "question": "Again?", "evidence": "", "SQL": "SELECT x FROM t", "difficulty": "challenging"},
    ]
    (root / "mini_dev_sqlite.json").write_text(json.dumps(questions))
    monkeypatch.setitem(eval_runner.SUITES, "bird_minidev", {**eval_runner.SUITES["bird_minidev"],
                        "path": root / "mini_dev_sqlite.json", "databases": root / "dev_databases"})
    return root


def bird_scout(db_folder, asked):
    """A stand-in Scout over one BIRD database: records what it was asked, answers with its value."""
    import sqlite3

    db = db_folder.name
    con = sqlite3.connect(db_folder / f"{db}.sqlite")
    rows = [list(r) for r in con.execute("SELECT x FROM t")]

    def interpret(question, history=(), question_id=None, use_saved=True):
        asked.append((db, question))
        return {"options": [{"meaning": "it", "choices": []}]}

    return types.SimpleNamespace(
        interpret=interpret,
        plan=lambda question, meaning, question_id=None: {"steps": ["Take it"]},
        run=lambda question, meaning, question_id=None, steps=None, save=True: {"rows": rows, "sql": "SELECT\n  x FROM t"},
        database=types.SimpleNamespace(run_all=lambda sql: (["x"], [list(r) for r in con.execute(sql)])),
        results=types.SimpleNamespace(get=lambda result_id: None),
    )


def test_bird_questions_come_from_birds_own_file(tmp_path, monkeypatch):
    fake_bird(tmp_path, monkeypatch)
    questions = load_questions("bird_minidev")
    assert [(q["n"], q["id"], q["db_id"]) for q in questions] == [(1, 101, "shop"), (2, 205, "zoo"), (3, 300, "shop")]
    assert questions[0]["gold_sql"] == "SELECT x FROM t" and questions[0]["evidence"] == "x is the count"


def test_a_bird_run_uses_one_database_per_question_and_writes_birds_format(store, tmp_path, monkeypatch):
    fake_bird(tmp_path, monkeypatch)
    asked, made = [], []

    def pipeline_for(folder):
        made.append(folder.name)
        return bird_scout(folder, asked)

    runner = EvalRunner(None, tmp_path / "runs", sleep=lambda s: None, pipeline_for=pipeline_for)
    progress = runner.start("bird_minidev", first=1, last=3, wait=True)

    assert progress["correct"] == 3 and progress["range"] == [1, 3] and progress["hints"] == "BIRD's evidence"
    assert made == ["shop", "zoo"]  # one Scout per database, made when first needed, then reused
    assert asked[0] == ("shop", "How many?\n(Hint: x is the count)")  # evidence as the hint
    assert all("SELECT x FROM t" not in question for _, question in asked)  # the gold SQL never reaches Scout
    predictions = json.loads((tmp_path / "runs" / "bird_predictions" / f"{progress['id']}.json").read_text())
    assert predictions == {
        "101": "SELECT x FROM t\t----- bird -----\tshop",
        "205": "SELECT x FROM t\t----- bird -----\tzoo",
        "300": "SELECT x FROM t\t----- bird -----\tshop",
    }
    assert [r["id"] for r in runner.finished_runs()] == [progress["id"]]  # the predictions file isn't read as a run


def test_a_range_picks_questions_first_to_last(store, tmp_path, monkeypatch):
    fake_bird(tmp_path, monkeypatch)
    asked = []
    runner = EvalRunner(None, tmp_path / "runs", sleep=lambda s: None, pipeline_for=lambda f: bird_scout(f, asked))
    progress = runner.start("bird_minidev", first=2, last=3, wait=True)
    assert progress["planned"] == [205, 300] and progress["range"] == [2, 3]
    for first, last in ((0, 2), (3, 2), (4, 0)):
        with pytest.raises(ValueError):
            runner.start("bird_minidev", first=first, last=last)


def test_fixes_from_bird_issues_are_never_applied(store, monkeypatch):
    from fastapi import HTTPException

    import backend.issues_api as issues_api

    monkeypatch.setattr(issues_api, "_store", lambda: store)
    issue_id = store.add_issue("eval", "How many?", {"suite": "bird_minidev", "db_id": "shop"}, run_id="r1")
    store.set_issue(issue_id, "report_ready", report={"fix_type": "error_book", "error_book_entry": {}})
    with pytest.raises(HTTPException) as refused:
        issues_api.approve_endpoint(issue_id)
    assert refused.value.status_code == 400 and "BIRD" in refused.value.detail
    assert store.issue(issue_id)["status"] == "report_ready"  # nothing applied


def test_bird_stays_out_of_what_normal_scout_sees():
    """Scout attaches every .sqlite under data/text_to_sql/: BIRD must live elsewhere."""
    from backend.config import BIRD_MINI_DEV_DIR, ROOT_DIR

    scout_folder = (ROOT_DIR / "data/text_to_sql").resolve()
    assert scout_folder not in BIRD_MINI_DEV_DIR.resolve().parents


def test_answers_are_scored_on_every_row_not_the_200_the_app_keeps(store, tmp_path):
    """BIRD #1506: Scout's SQL (no DISTINCT) gave 933 rows holding all 21 products; the first 200
    held only 4 of them, so comparing the kept rows scored a right answer wrong."""
    import sqlite3

    from backend.eval_runner import as_set

    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE sales (product TEXT)")
    con.executemany("INSERT INTO sales VALUES (?)", [("Nafta",)] * 300 + [("LPG",), ("Natural",)])
    gold_sql, scout_sql = "SELECT DISTINCT product FROM sales", "SELECT product FROM sales"
    kept = [list(r) for r in con.execute(scout_sql).fetchmany(200)]  # what the app keeps: only 'Nafta'
    assert as_set(kept) != as_set(con.execute(gold_sql).fetchall())

    pipeline = types.SimpleNamespace(
        database=types.SimpleNamespace(run_all=lambda sql: (["product"], [list(r) for r in con.execute(sql)])),
        results=types.SimpleNamespace(get=lambda result_id: None),
    )
    runner = EvalRunner(pipeline, tmp_path / "runs", sleep=lambda s: None)
    runner.run = {"id": "r", "suite": "practice"}
    q = {"id": 1506, "db_id": "retail_store", "question": "Products sold?", "gold_sql": gold_sql}
    entry = runner._score(q, "eval:r:1506", {"rows": kept[:10], "sql": scout_sql}, "", [], 1.0)
    assert entry["correct"] and entry["outcome"] == "right"


def test_one_question_can_be_checked_again_without_touching_the_runs(store, tmp_path):
    """Version-21: Issues -> Run this question again. One question by its id, saved in runs/checks/:
    not a run (scores, run list unchanged) and a wrong answer doesn't log another issue."""
    seen = []
    runner = EvalRunner(fake_pipeline([{"rows": [[99]], "sql": "SELECT 99"}], store, seen), tmp_path / "runs",
                        sleep=lambda s: None)
    second = QUESTIONS[1]
    runner.check("practice", second["id"], wait=True)

    assert not list((tmp_path / "runs").glob("*.json"))  # no run file: the Evaluation tab is unchanged
    assert runner.progress() is None and runner.finished_runs() == []
    checks = runner.checks(second["id"])
    assert len(checks) == 1 and checks[0]["status"] == "finished" and checks[0]["range"] == [2, 2]
    result = checks[0]["results"][0]
    assert result["id"] == second["id"] and result["outcome"] == "wrong rows" and result["llm_calls"] == 3
    assert store.issues(scope=["eval"])["total"] == 0  # already an issue: not logged again
    assert runner.checks(QUESTIONS[0]["id"]) == []
    with pytest.raises(ValueError):
        runner.check("practice", 999999)


def test_corrected_bird_takes_its_own_questions_and_answers_and_birds_difficulty(tmp_path, monkeypatch):
    """Version-21: the corrected Mini-Dev (string ids, no difficulty) on BIRD's databases with other
    descriptions; a Scout per suite and database, since the two suites share database names."""
    from backend import eval_runner

    root = fake_bird(tmp_path, monkeypatch)
    fixed = [{"question_id": "205", "db_id": "zoo", "question": "Which animal, by name?", "evidence": "",
              "SQL": "SELECT x + 1 FROM t", "original_SQL": "SELECT x FROM t"}]
    (tmp_path / "fixed.json").write_text(json.dumps(fixed))
    monkeypatch.setitem(eval_runner.SUITES, "bird_corrected", {
        **eval_runner.SUITES["bird_corrected"], "path": tmp_path / "fixed.json",
        "databases": root / "dev_databases", "difficulty_from": root / "mini_dev_sqlite.json"})
    questions = load_questions("bird_corrected")
    assert questions == [{"n": 1, "id": 205, "db_id": "zoo", "question": "Which animal, by name?", "evidence": "",
                          "gold_sql": "SELECT x + 1 FROM t", "difficulty": "moderate"}]
    made = []
    runner = EvalRunner(None, tmp_path / "runs", pipeline_for=lambda folder: made.append(folder) or object())
    assert runner.pipeline_for_db("bird_minidev", "zoo") is not runner.pipeline_for_db("bird_corrected", "zoo")
    runner.pipeline_for_db("bird_corrected", "zoo")
    assert len(made) == 2  # one each, then reused
    assert eval_runner.is_benchmark("bird_corrected")  # fixes from its issues are never applied either


def test_a_crash_on_one_question_fails_that_question_not_the_run(store, tmp_path):
    """Version-21: BIRD (corrected) #1317 stopped a 50-question run at 31: Scout wrote 'Women's Soccer'
    (an unclosed quote), and sqlglot's TokenError wasn't caught. Now the question is "no answer"."""
    seen = []
    pipeline = fake_pipeline([{"rows": [[1]], "sql": "SELECT 1"}], store, seen)

    def run(question, meaning, question_id=None, steps=None, save=True):
        raise RuntimeError("Error tokenizing ' 'Women's Soccer'")

    answers = iter([run, None])
    real_run = pipeline.run
    pipeline.run = lambda *a, **k: (next(answers) or real_run)(*a, **k)
    runner = EvalRunner(pipeline, tmp_path / "runs", sleep=lambda s: None)
    progress = runner.start("practice", last=2, wait=True)
    assert progress["status"] == "finished" and progress["done"] == 2
    first = json.loads((tmp_path / "runs" / f"{progress['id']}.json").read_text())["results"][0]
    assert first["outcome"] == "no answer" and "Women's Soccer" in first["error"]


def test_an_unclosed_quote_is_an_sql_problem_not_a_crash():
    """sqlglot raises TokenError (not ParseError) on an unclosed quote: every check now handles it."""
    from backend.text_to_sql.checks import merged_records, tables_in_sql
    from backend.text_to_sql.safety import write_problems

    sql = "SELECT COUNT(*) FROM event e WHERE e.event_name = 'Women's Soccer'\n  AND e.size = 'Medium'"
    assert tables_in_sql(sql, ["event"]) == []
    assert merged_records(sql, [], lambda s: ([], [])) == []
    assert write_problems(sql) == []  # unreadable, but no write keyword: SQLite decides if it's valid
    assert write_problems("DELETE FROM event WHERE name = 'Women's'") == ["DELETE"]
