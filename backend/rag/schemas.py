from typing import List, Dict, Optional
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    id: str
    score: float
    source: str
    snippet: str
    text: Optional[str] = None  # the whole chunk


class QueryResponse(BaseModel):
    answer: str
    citations: List[RetrievedChunk]
    reasoning_summary: Optional[str] = None