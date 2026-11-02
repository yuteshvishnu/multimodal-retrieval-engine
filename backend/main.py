from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.rag.api import router as rag_router
from backend.text_to_sql.api import router as sql_router

app = FastAPI(title="Rexie")

# RAG: /api/rag/...   Text-to-SQL: /api/sql/...
app.include_router(rag_router)
app.include_router(sql_router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# serve the built React app (frontend/dist), if it has been built
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
