from typing import Any, List, Optional
from pydantic import BaseModel


class QueryResponse(BaseModel):
    sql: Optional[str] = None
    tables: List[str] = []  # tables the answer used, read from its SQL (on failure: the tables offered)
    tables_offered: List[str] = []  # tables the search picked and sent to the LLM
    columns: List[str] = []
    rows: List[List[Any]] = []
    error: Optional[str] = None
    confidence: Optional[str] = None  # high / medium / low (Version-4)
    warnings: List[str] = []          # results of the deterministic checks
    llm_calls: int = 0
    attempts: List[dict] = []         # every SQL the model wrote, and what happened to it
    feedback: Optional[str] = None    # what the model was told when its attempts disagreed
    meaning: Optional[str] = None     # the meaning the user confirmed (Version-5)
    safety: Optional[str] = None      # set when the request was blocked as a possible write (Version-7)
    limit: Optional[str] = None       # set when the query was too big or too slow (Version-8)
    failure_reason: Optional[str] = None  # the raw error when no query worked (Version-9)
    row_count: int = 0                # rows kept (at most MAX_RESULT_ROWS); `rows` is only page 1
    more: bool = False                # there were more rows than MAX_RESULT_ROWS
    page_size: int = 0
    result_id: Optional[str] = None   # for loading other pages
    question_id: Optional[str] = None  # the question's id on the Dashboard
    steps: List[str] = []             # the steps the user confirmed (Version-10)
    reused: bool = False              # answered by re-running a saved answer's SQL (Version-12)
    fallback: bool = False            # the saved answer couldn't be reused: write the steps as usual


class InterpretResponse(BaseModel):
    options: List[dict] = []          # up to 3 meanings: {"meaning": ..., "choices": [...]}
    round: int = 0
    max_rounds: int = 0
    tables: List[str] = []
    llm_calls: int = 0
    error: Optional[str] = None
    safety: Optional[str] = None
    question_id: Optional[str] = None


class PlanResponse(BaseModel):
    steps: List[str] = []             # the plan, for the user to check (Version-10)
    round: int = 0
    max_rounds: int = 0
    tables: List[str] = []
    llm_calls: int = 0
    error: Optional[str] = None
    safety: Optional[str] = None
    question_id: Optional[str] = None
