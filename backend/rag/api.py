from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.rag.schemas import QueryResponse

from backend.rag.documents import MAX_UPLOAD_BYTES, DocumentStore
from backend.rag.pipeline import RAGPipeline
from backend.usage import Timer, usage_store
from backend.rag.feedback.logger import log_feedback

router = APIRouter(prefix="/api/rag", tags=["RAG"])

class FeedbackPayload(BaseModel):
    query_text: str
    answer: str
    citations: list[dict]
    rating: str  # "up" or "down"
    comment: str | None = None
    metadata: dict | None = None
    question_id: str | None = None  # Scribe's question on the Dashboard

# single pipeline instance for the whole app
pipeline = RAGPipeline()
documents = DocumentStore(pipeline.index, pipeline.text_encoder)


@router.post("/query", response_model=QueryResponse)
def query_endpoint(
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

    with Timer(usage_store) as timer:
        timer.question_id = usage_store.start_question("scribe", query_text or "")
        result = pipeline.run(
            query_text=query_text,
            sources=source_list,
            collections=collection_list,
            question_id=timer.question_id,
        )
        usage_store.finish(timer.question_id, "answered" if result.get("citations") else "no matching passages")

    return JSONResponse({**result, "question_id": timer.question_id})

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
    if payload.question_id:
        try:
            usage_store.add_feedback(payload.question_id, rating, payload.comment or "", {
                "answer": payload.answer, "citations": payload.citations,
            })
        except KeyError:
            pass  # an answer from before the Dashboard existed

    return {"status": "ok"}


# ---------- Scribe Version-2: the user's sources ----------

@router.get("/documents")
def documents_endpoint(page: int = 1, search: str = "", all: bool = False):
    """Every document Scribe can answer from: 10 per page, or all names (for the picker)."""
    if all:
        return {"items": [{"name": c["name"], "type": c["type"]} for c in documents.sources()]}
    return documents.list(page, search)


@router.get("/documents/{source}/chunks")
def chunks_endpoint(source: str, page: int = 1):
    """One page (10) of the passages a document was split into."""
    try:
        return documents.chunks(source, page)
    except KeyError:
        raise HTTPException(404, f"No document called '{source}'.")


@router.delete("/documents/{source}")
def delete_document_endpoint(source: str):
    """Removes a document from Scribe: its file and its passages in the index."""
    try:
        documents.delete(source)
    except KeyError:
        raise HTTPException(404, f"No document called '{source}'.")
    return {"deleted": source}


@router.post("/documents/upload")
def upload_endpoint(file: UploadFile = File(...)):
    """Adds a .txt, .md, .pdf or .docx file: chunks it, embeds it, and makes it searchable at once."""
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    try:
        card = documents.add(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"document": card}
