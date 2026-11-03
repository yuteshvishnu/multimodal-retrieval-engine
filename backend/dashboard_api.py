"""
The Dashboard's data: totals, cost per hour / day / month for a chosen period, and every question with its
metrics (LLM calls per step, retries, tokens, time, outcome, cost). Read from backend/usage.py.
"""
from datetime import date

from fastapi import APIRouter, HTTPException

from backend.config import LLM_PRICES
from backend.usage import usage_store

router = APIRouter(prefix="/api/usage", tags=["Dashboard"])


@router.get("/summary")
def summary_endpoint():
    """Questions asked, LLM calls, input / output tokens, total cost, and the prices used."""
    return {**usage_store.summary(), "prices": {m: {"input": p[0], "output": p[1]} for m, p in LLM_PRICES.items()}}


@router.get("/costs")
def costs_endpoint(period: str = "day", day: str = ""):
    """LLM cost per hour (day), day (week, month) or month (year), for the period containing `day`."""
    try:
        when = date.fromisoformat(day) if day else date.today()
        return usage_store.costs(period, when)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/questions")
def questions_endpoint(page: int = 1, search: str = "", feedback: str = ""):
    """Every question, newest first, 10 per page, with its metrics. feedback=down: only the 👎 (issues)."""
    return usage_store.questions(page, search, feedback)
