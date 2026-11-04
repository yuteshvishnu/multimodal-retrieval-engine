"""
Version-11: the Dashboard's Issues view. Scout's failures (eval, user reports, SQL errors)
and the Failure Analyst agent that investigates them.

    GET  /api/issues                    issues, 10 per page, filter by source / status; scope = the
                                        sources a view shows (Dashboard: user_report,sql_error; Evaluation: eval);
                                        run = one evaluation run's issues (Version-15; "none": logged without a run)
    GET  /api/issues/{id}               one issue, with every step the agent took (polled while it runs)
    POST /api/issues/{id}/investigate   starts the agent on it, in the background
    POST /api/issues/investigate-new    starts it on the oldest new issues (AGENT_BATCH at most)
    POST /api/issues/{id}/approve       applies the suggested fix (error book / description) and closes it
    POST /api/issues/{id}/reject        closes it without changing anything

The agent runs only when the developer clicks, one issue at a time (rate limits), in a
background thread, so the page stays responsive and shows each step as it happens.
"""
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Form, HTTPException

from backend import usage
from backend.agent.analyst import FailureAnalyst
from backend.agent.fixes import apply_fix
from backend.config import AGENT_BATCH, AGENT_MODEL, AGENT_MAX_TOOL_CALLS, LLM_API_KEY
from backend.eval_runner import is_benchmark
from backend.text_to_sql.api import pipeline

router = APIRouter(prefix="/api/issues", tags=["Issues"])

# fixes that change Scout when approved; never applied from a benchmark's (BIRD's) issues
BENCHMARK_BLOCKED = ("error_book", "table_description")

# one investigation at a time; the rest wait their turn
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="failure-analyst")


def _store():
    return usage.usage_store  # looked up each time, so tests can swap it


def _issue(issue_id: str) -> dict:
    issue = _store().issue(issue_id)
    if issue is None:
        raise HTTPException(404, "Unknown issue.")
    return issue


def _pipeline_for(issue: dict):
    """The Scout the issue happened on: the app's, or for a BIRD question one with only its database."""
    details = issue.get("details") or {}
    if not is_benchmark(details.get("suite", "")):
        return pipeline
    from backend.eval_api import runner  # the runner keeps one Scout per BIRD database
    return runner.pipeline_for_db(details["suite"], details["db_id"])


def _start(issue_id: str) -> bool:
    store = _store()
    try:
        scout = _pipeline_for(_issue(issue_id))
    except RuntimeError:
        return False  # its BIRD database isn't on disk any more
    if not store.start_investigation(issue_id):
        return False
    _worker.submit(FailureAnalyst(scout, store).investigate, issue_id)
    return True


def _scope(text: str) -> list:
    return [s for s in (text or "").split(",") if s in usage.ISSUE_SOURCES]


@router.get("")
def issues_endpoint(page: int = 1, source: str = "", status: str = "", scope: str = "", run: str = ""):
    return {**_store().issues(page, source, status, _scope(scope), run or None), "agent": {
        "model": AGENT_MODEL, "max_tool_calls": AGENT_MAX_TOOL_CALLS, "batch": AGENT_BATCH, "ready": bool(LLM_API_KEY),
    }}


@router.get("/{issue_id}")
def issue_endpoint(issue_id: str):
    return _issue(issue_id)


@router.post("/investigate-new")
def investigate_new_endpoint(scope: str = Form(""), run: str = Form("")):
    if not LLM_API_KEY:
        raise HTTPException(503, "The LLM API key is not set (LLM_PROVIDER and its key in .env), so the agent can't run.")
    started = [i for i in _store().new_issue_ids(AGENT_BATCH, _scope(scope), run or None) if _start(i)]
    return {"started": started}


@router.post("/{issue_id}/investigate")
def investigate_endpoint(issue_id: str):
    _issue(issue_id)
    if not LLM_API_KEY:
        raise HTTPException(503, "The LLM API key is not set (LLM_PROVIDER and its key in .env), so the agent can't run.")
    if not _start(issue_id):
        raise HTTPException(409, "The agent is already investigating this issue.")
    return {"started": [issue_id]}


@router.post("/{issue_id}/approve")
def approve_endpoint(issue_id: str, note: str = Form("")):
    issue = _issue(issue_id)
    if issue["status"] != "report_ready":
        raise HTTPException(400, "Only a finished report can be approved.")
    changes_scout = issue["report"].get("fix_type") in BENCHMARK_BLOCKED
    if changes_scout and is_benchmark((issue.get("details") or {}).get("suite", "")):
        # Version-18: a fix learned from a benchmark's answers would teach Scout the test (and a
        # description note would be written into BIRD's own files)
        raise HTTPException(400, "This fix can't be approved for a BIRD question: it comes from BIRD's correct "
                                 "answer, so applying it would inflate BIRD scores. Note what went wrong and fix it "
                                 "in general instead.")
    try:
        applied = apply_fix(issue, pipeline)
    except ValueError as e:
        raise HTTPException(400, f"Could not apply the fix: {e}")
    _store().set_issue(issue_id, "approved", report={**issue["report"], "applied": applied}, note=note)
    return _issue(issue_id)


@router.post("/{issue_id}/reject")
def reject_endpoint(issue_id: str, note: str = Form("")):
    issue = _issue(issue_id)
    if issue["status"] == "investigating":
        raise HTTPException(400, "Wait for the agent to finish first.")
    _store().set_issue(issue_id, "rejected", note=note)
    return _issue(issue_id)
