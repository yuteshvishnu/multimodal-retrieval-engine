from typing import Any, List, Optional
from pydantic import BaseModel


class QueryResponse(BaseModel):
    sql: Optional[str] = None
    tables: List[str] = []  # tables sent to the LLM
    columns: List[str] = []
    rows: List[List[Any]] = []
    error: Optional[str] = None
