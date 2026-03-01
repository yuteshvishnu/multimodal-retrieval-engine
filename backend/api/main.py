from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from backend.api.schemas import QueryResponse

from backend.core.pipeline import MultimodalPipeline  # 👈 NEW

app = FastAPI()

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
    query_text: str = Form(...),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    sources: str | None = Form(None),
):
    """
    Now:
    - read file bytes (if present)
    - pass everything to the multimodal pipeline
    - return pipeline result
    """

    image_bytes = await image.read() if image is not None else None
    audio_bytes = await audio.read() if audio is not None else None

    source_list: list[str] | None = None
    if sources:
        # e.g. "heart_notes, lungs_intro"
        source_list = [s.strip() for s in sources.split(",") if s.strip()]

    result = pipeline.run(
        query_text=query_text,
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        sources=source_list,
    )

    # for now pipeline returns a plain dict already
    return JSONResponse(result)