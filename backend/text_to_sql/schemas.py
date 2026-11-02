from typing import Any, List, Optional
from pydantic import BaseModel


class QueryResponse(BaseModel):
    sql: Optional[str] = None
    tables: List[str] = []  # tables sent to the LLM
    columns: List[str] = []
    rows: List[List[Any]] = []
    error: Optional[str] = None
    confidence: Optional[str] = None  # high / medium / low (step 4)
    warnings: List[str] = []          # results of the deterministic checks
    llm_calls: int = 0
    attempts: List[dict] = []         # every SQL the model wrote, and what happened to it
    feedback: Optional[str] = None    # what the model was told when its attempts disagreed
    meaning: Optional[str] = None     # the meaning the user confirmed (step 5)


class InterpretResponse(BaseModel):
    options: List[dict] = []          # up to 3 meanings: {"meaning": ..., "choices": [...]}
    round: int = 0
    max_rounds: int = 0
    tables: List[str] = []
    llm_calls: int = 0
    error: Optional[str] = None
