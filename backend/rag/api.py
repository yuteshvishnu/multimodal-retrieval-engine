from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.rag.schemas import QueryResponse

from backend.rag.pipeline import RAGPipeline
from backend.rag.feedback.logger import log_feedback

router = APIRouter(prefix="/api/rag", tags=["Version-1: RAG"])

class FeedbackPayload(BaseModel):
    query_text: str
    answer: str
    citations: list[dict]
    rating: str  # "up" or "down"
    comment: str | None = None
    metadata: dict | None = None

# single pipeline instance for the whole app
pipeline = RAGPipeline()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    query_text: str | None = Form(None),
    sources: str | None = Form(None),
    collections: str | None = Form(None),
):
    source_list: list[str] | None = None
    if sources:
        source_list = [s.strip() for s in sources.split(",") if s.strip()]

    collection_list: list[str] | None = None
    if collections:
        collection_list = [c.strip() for c in collections.split(",") if c.strip()]

    result = pipeline.run(
        query_text=query_text,
        sources=source_list,
        collections=collection_list,
    )

    return JSONResponse(result)

@router.post("/feedback")
async def feedback_endpoint(payload: FeedbackPayload):
    """
    Accepts feedback on a given answer/citations and logs it to disk.
    """
    # Basic validation of rating
    rating = payload.rating.lower()
    if rating not in ("up", "down"):
        return JSONResponse(
            {"status": "error", "message": "rating must be 'up' or 'down'"},
            status_code=400,
        )

    event = {
        "query_text": payload.query_text,
        "answer": payload.answer,
        "citations": payload.citations,
        "rating": rating,
        "comment": payload.comment,
        "metadata": payload.metadata or {},
    }

    log_feedback(event)

    return {"status": "ok"}
