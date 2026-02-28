from typing import List, Dict, Optional
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    id: str
    score: float
    source: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[RetrievedChunk]
    reasoning_summary: Optional[str] = None
    used_image: bool = False