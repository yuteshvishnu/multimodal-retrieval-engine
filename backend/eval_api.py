"""
Evaluation (Version-13): how accurate Scout is on a benchmark, for the Evaluation tab and the
ring next to Rexie's name.

    GET /api/eval/summary   the latest run: accuracy, by difficulty, by outcome, LLM calls and cost
    GET /api/eval/results   its questions, 10 per page; filter right / wrong (scout / doubtful), by difficulty, search

Running one from the tab (Version-14, backend/eval_runner.py):

    GET  /api/eval/suites              the test suites, and an estimate for a run of n questions
    POST /api/eval/runs                start a run (suite, first, last: questions first..last); one at a time
    GET  /api/eval/runs/current        the run's progress, polled while it runs
    GET  /api/eval/runs                finished runs, newest first, with their issue counts (Version-15:
                                       Evaluation -> Issues picks a run, and the agent works on its issues)
    POST /api/eval/runs/{id}/stop      stop after the current question
    POST /api/eval/runs/{id}/resume    carry on from where it stopped
    POST /api/eval/checks              Version-21: run one question again (suite, question_id), from its issue
    GET  /api/eval/checks              that question's checks, newest first, and the one running now

Comparing flows (Version-22): each run records its flow (Original, Full, Adaptive).

    GET  /api/eval/compare             each flow's score on the same questions (suite, first, last; last 0 = to the end)
    POST /api/eval/comparisons         run several flows on those questions, one after another (suite, first, last,
                                       flows, reuse: skip a flow that already has a run covering them)
    GET  /api/eval/comparisons/current the comparison being run, or the last one
    POST /api/eval/comparisons/stop    stop after the current question
    POST /api/eval/comparisons/resume  carry on from where it stopped

A run is one JSON file: {"benchmark", "run_at", "model", "mode", "results": [one per question]}.
The newest file in data/text_to_sql/eval/runs/ is shown; until a real run exists, a sample run
(data/text_to_sql/eval/sample_bird_run.json, made-up questions in BIRD's format, "sample": true)
is shown instead, and the UI says so. Totals are always worked out from the results, so they
can't disagree with them. Wrong answers link to their issue (Evaluation -> Issues) when the eval
logged one.
"""
import json
import math
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Form, HTTPException

from backend import usage
from backend.config import LLM_MODEL_SQL
from backend.eval_runner import DEFAULT_FLOW, FLOWS, EvalRunner, estimate, flow_of, flows, models, suites
from backend.text_to_sql.api import pipeline
from backend.text_to_sql.pipeline import TextToSQLPipeline

router = APIRouter(prefix="/api/eval", tags=["Evaluation"])

EVAL_DIR = Path("data/text_to_sql/eval")
RUNS_DIR = EVAL_DIR / "runs"
SAMPLE_RUN = EVAL_DIR / "sample_bird_run.json"
# Version-19: BIRD answers we checked by hand and believe are wrong (labels only, never the score)
DOUBTFUL_FILE = EVAL_DIR / "bird_doubtful.json"
PAGE_SIZE = 10
DIFFICULTIES = ("simple", "moderate", "challenging")


def latest_run() -> Dict:
    """The newest real run that isn't still running and has answers, else the sample run."""
    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True) if RUNS_DIR.exists() else []:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") != "running" and run.get("results"):
            return run
    if not SAMPLE_RUN.exists():
        return {"benchmark": "BIRD dev", "short": "BIRD", "sample": False, "results": []}
    return json.loads(SAMPLE_RUN.read_text(encoding="utf-8"))


def doubtful_answers() -> Dict[str, Dict]:
    """BIRD question_id -> {"reason", "checked"}; read each time, so an edit shows at once."""
    if not DOUBTFUL_FILE.exists():
        return {}
    return json.loads(DOUBTFUL_FILE.read_text(encoding="utf-8")).get("questions", {})


def label_doubtful(run: Dict) -> List[Dict]:
    """The run's results; a wrong BIRD answer we believe BIRD got wrong carries "doubtful": reason.
    It still counts as wrong: only the label is added."""
    results = run.get("results", [])
    if run.get("suite") != "bird_minidev":
        return results
    doubtful = doubtful_answers()
    return [
        {**r, "doubtful": doubtful[str(r.get("id"))]["reason"]}
        if not r.get("correct") and str(r.get("id")) in doubtful else r
        for r in results
    ]


def score(results: List[Dict]) -> Dict:
    right = sum(1 for r in results if r.get("correct"))
    return {"total": len(results), "correct": right,
            "accuracy": round(100 * right / len(results), 1) if results else None}


def summarize(run: Dict) -> Dict:
    results = label_doubtful(run)
    outcomes: Dict[str, int] = {}
    for r in results:
        outcomes[r.get("outcome", "wrong rows")] = outcomes.get(r.get("outcome", "wrong rows"), 0) + 1
    calls = sum(r.get("llm_calls", 0) for r in results)
    return {
        **{k: run.get(k) for k in ("suite", "benchmark", "note", "run_at", "model", "provider", "mode", "status")},
        "flow": FLOWS.get(flow_of(run), {}).get("name"),  # Version-22
        "short": run.get("short") or "BIRD",
        "sample": bool(run.get("sample")),
        **score(results),
        "by_difficulty": {d: score([r for r in results if r.get("difficulty") == d]) for d in DIFFICULTIES},
        "by_outcome": outcomes,
        "doubtful": sum(1 for r in results if r.get("doubtful")),  # wrong, but BIRD's answer looks wrong
        "databases": len({r.get("db_id") for r in results}),
        "llm_calls": calls,
        "avg_llm_calls": round(calls / len(results), 1) if results else 0,
        "cost": round(sum(r.get("cost", 0) for r in results), 5),
    }


@router.get("/summary")
def summary_endpoint():
    return summarize(latest_run())


@router.get("/results")
def results_endpoint(page: int = 1, filter: str = "all", difficulty: str = "", search: str = ""):
    latest = latest_run()
    results = label_doubtful(latest)
    if filter in ("right", "wrong"):
        results = [r for r in results if bool(r.get("correct")) == (filter == "right")]
    elif filter == "scout":  # wrong, and not a doubtful BIRD answer: what's worth fixing
        results = [r for r in results if not r.get("correct") and not r.get("doubtful")]
    elif filter == "doubtful":
        results = [r for r in results if r.get("doubtful")]
    if difficulty in DIFFICULTIES:
        results = [r for r in results if r.get("difficulty") == difficulty]
    if search.strip():
        needle = search.strip().lower()
        results = [r for r in results if needle in r.get("question", "").lower() or needle in r.get("db_id", "").lower()]
    pages = max(1, math.ceil(len(results) / PAGE_SIZE))
    page = min(max(1, page), pages)
    items = results[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    run_id = None if latest.get("sample") else latest.get("id")
    issues = usage.usage_store.issue_ids_for("eval", [r["question"] for r in items if not r.get("correct")], run_id)
    return {
        "items": [{**r, "issue": issues.get(r["question"])} for r in items],
        "page": page, "pages": pages, "total": len(results),
    }


# ---------- Version-14: run an evaluation and watch it ----------

def bird_pipeline(folder: Path) -> TextToSQLPipeline:
    """Version-18: a Scout that sees only this one BIRD database, given its whole schema (BIRD's
    databases have 3-13 tables, so no table search and no LLM-written table summaries are needed)."""
    return TextToSQLPipeline(use_retrieval=False, data_dir=str(folder))


runner = EvalRunner(pipeline, pipeline_for=bird_pipeline, practice_for=TextToSQLPipeline)


def _runner() -> EvalRunner:
    return runner  # looked up each time, so tests can swap it


@router.get("/suites")
def suites_endpoint(questions: int = 0, flows_to_run: str = "", model: str = LLM_MODEL_SQL):
    # Version-22: flows_to_run = "original,full": the estimate for running each of them on those questions
    chosen = [f for f in flows_to_run.split(",") if f in FLOWS] or None
    return {"suites": suites(), "flows": flows(), "models": models(), "default_model": LLM_MODEL_SQL,
            "estimate": estimate(questions, chosen, model) if questions > 0 else None,
            "running": _runner().busy()}


@router.post("/runs")
def start_endpoint(suite: str = Form(...), first: int = Form(1), last: int = Form(0), flow: str = Form(DEFAULT_FLOW),
                   model: str = Form(LLM_MODEL_SQL)):
    try:
        return _runner().start(suite, first, last, flow=flow, model=model)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/runs")
def runs_endpoint():
    runs = _runner().finished_runs()
    counts = usage.usage_store.issue_counts_by_run()
    return {
        "runs": [{**r, "issues": counts.get(r["id"], {"total": 0, "new": 0})} for r in runs],
        # eval issues logged without a run (from the command-line eval script)
        "unlinked": counts.get("none", {"total": 0, "new": 0}),
    }


@router.get("/runs/current")
def current_endpoint():
    return {"run": _runner().progress()}


@router.post("/runs/{run_id}/stop")
def stop_endpoint(run_id: str):
    r = _runner()
    if not r.busy() or (r.run or {}).get("id") != run_id:
        raise HTTPException(400, "That run isn't running.")
    r.stop()
    return {"stopping": True}


@router.post("/runs/{run_id}/resume")
def resume_endpoint(run_id: str):
    try:
        return _runner().resume(run_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------- Version-21: run one question again, from its issue ----------

def _check_view(run: Dict) -> Dict:
    result = (label_doubtful(run) or [None])[0]
    return {k: run.get(k) for k in ("id", "status", "message", "run_at", "finished_at", "model", "current")} | {"result": result}


@router.post("/checks")
def start_check_endpoint(suite: str = Form(...), question_id: str = Form(...)):
    try:
        run = _runner().check(suite, question_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": run["id"], "status": run["status"]}


@router.get("/checks")
def checks_endpoint(question_id: str):
    r = _runner()
    live = r.progress() if r.busy() else None
    running = live if live and live.get("check") and str(live.get("question_id")) == str(question_id) else None
    return {
        "running": {k: running.get(k) for k in ("id", "status", "current", "run_at")} if running else None,
        "busy": r.busy() and running is None,  # another run or check is using the runner
        "checks": [_check_view(run) for run in r.checks(question_id)[:5]],
    }


# ---------- Version-22: compare flows on the same questions ----------

@router.get("/compare")
def compare_endpoint(suite: str, first: int = 1, last: int = 0, model: str = LLM_MODEL_SQL):
    try:
        return _runner().compare(suite, first, last, model)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/comparisons")
def start_comparison_endpoint(suite: str = Form(...), first: int = Form(1), last: int = Form(0), flows: str = Form(...),
                              reuse: bool = Form(True), model: str = Form(LLM_MODEL_SQL)):
    try:
        return _runner().start_comparison(suite, first, last, [f for f in flows.split(",") if f], reuse, model=model)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/comparisons/current")
def current_comparison_endpoint():
    r = _runner()
    return {"comparison": r.comparison(), "busy": r.busy()}


@router.post("/comparisons/stop")
def stop_comparison_endpoint():
    r = _runner()
    if not r.busy():
        raise HTTPException(400, "No comparison is running.")
    r.stop()
    return {"stopping": True}


@router.post("/comparisons/resume")
def resume_comparison_endpoint():
    try:
        return _runner().resume_comparison()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
