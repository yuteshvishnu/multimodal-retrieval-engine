from typing import Any, List, Optional
from pydantic import BaseModel


class QueryResponse(BaseModel):
    sql: Optional[str] = None
    tables: List[str] = []  # tables sent to the LLM
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
    row_count: int = 0                # rows kept (at most MAX_RESULT_ROWS); `rows` is only page 1
    more: bool = False                # there were more rows than MAX_RESULT_ROWS
    page_size: int = 0
    result_id: Optional[str] = None   # for loading other pages


class InterpretResponse(BaseModel):
    options: List[dict] = []          # up to 3 meanings: {"meaning": ..., "choices": [...]}
    round: int = 0
    max_rounds: int = 0
    tables: List[str] = []
    llm_calls: int = 0
    error: Optional[str] = None
    safety: Optional[str] = None
