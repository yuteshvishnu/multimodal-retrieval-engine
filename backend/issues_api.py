"""
Version-11: the Dashboard's Issues view. Scout's failures (eval, user reports, SQL errors)
and the Failure Analyst agent that investigates them.

    GET  /api/issues                    issues, 10 per page, filter by source / status
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
from backend.config import AGENT_BATCH, AGENT_MODEL, AGENT_MAX_TOOL_CALLS, GROQ_API_KEY
from backend.text_to_sql.api import pipeline

router = APIRouter(prefix="/api/issues", tags=["Issues"])

# one investigation at a time; the rest wait their turn
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="failure-analyst")


def _store():
    return usage.usage_store  # looked up each time, so tests can swap it


def _issue(issue_id: str) -> dict:
    issue = _store().issue(issue_id)
    if issue is None:
        raise HTTPException(404, "Unknown issue.")
    return issue


def _start(issue_id: str) -> bool:
    store = _store()
    if not store.start_investigation(issue_id):
        return False
    _worker.submit(FailureAnalyst(pipeline, store).investigate, issue_id)
    return True


@router.get("")
def issues_endpoint(page: int = 1, source: str = "", status: str = ""):
    return {**_store().issues(page, source, status), "agent": {
        "model": AGENT_MODEL, "max_tool_calls": AGENT_MAX_TOOL_CALLS, "batch": AGENT_BATCH, "ready": bool(GROQ_API_KEY),
    }}


@router.get("/{issue_id}")
def issue_endpoint(issue_id: str):
    return _issue(issue_id)


@router.post("/investigate-new")
def investigate_new_endpoint():
    if not GROQ_API_KEY:
        raise HTTPException(503, "GROQ_API_KEY is not set, so the agent can't run.")
    started = [i for i in _store().new_issue_ids(AGENT_BATCH) if _start(i)]
    return {"started": started}


@router.post("/{issue_id}/investigate")
def investigate_endpoint(issue_id: str):
    _issue(issue_id)
    if not GROQ_API_KEY:
        raise HTTPException(503, "GROQ_API_KEY is not set, so the agent can't run.")
    if not _start(issue_id):
        raise HTTPException(409, "The agent is already investigating this issue.")
    return {"started": [issue_id]}


@router.post("/{issue_id}/approve")
def approve_endpoint(issue_id: str, note: str = Form("")):
    issue = _issue(issue_id)
    if issue["status"] != "report_ready":
        raise HTTPException(400, "Only a finished report can be approved.")
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
