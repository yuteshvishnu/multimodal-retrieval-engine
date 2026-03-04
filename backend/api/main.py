from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from backend.api.schemas import QueryResponse

from backend.core.pipeline import MultimodalPipeline  # 👈 NEW
from backend.feedback.logger import log_feedback

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

class FeedbackPayload(BaseModel):
    query_text: str
    answer: str
    citations: list[dict]
    rating: str  # "up" or "down"
    comment: str | None = None
    metadata: dict | None = None

templates = Jinja2Templates(directory="frontend/templates")

# single pipeline instance for the whole app
pipeline = MultimodalPipeline()  # 👈 NEW


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    query_text: str | None = Form(None),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    sources: str | None = Form(None),
    collections: str | None = Form(None),
):
    image_bytes = await image.read() if image is not None else None
    audio_bytes = await audio.read() if audio is not None else None

    source_list: list[str] | None = None
    if sources:
        source_list = [s.strip() for s in sources.split(",") if s.strip()]

    collection_list: list[str] | None = None
    if collections:
        collection_list = [c.strip() for c in collections.split(",") if c.strip()]

    result = pipeline.run(
        query_text=query_text,
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        sources=source_list,
        collections=collection_list,
    )

    return JSONResponse(result)

@app.post("/feedback")
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