from typing import Any, List, Optional
from pydantic import BaseModel


class QueryResponse(BaseModel):
    sql: Optional[str] = None
    columns: List[str] = []
    rows: List[List[Any]] = []
    error: Optional[str] = None
