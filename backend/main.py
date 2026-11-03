import base64
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from backend.config import APP_PASSWORD
from backend.dashboard_api import router as dashboard_router
from backend.issues_api import router as issues_router
from backend.rag.api import router as rag_router
from backend.text_to_sql.api import router as sql_router

app = FastAPI(title="Rexie")


@app.middleware("http")
async def require_password(request: Request, call_next):
    """
    Optional login (APP_PASSWORD in .env): HTTP Basic auth, so the browser shows its own
    sign-in box once and then sends the password with every request (UI and API).
    """
    if not APP_PASSWORD:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    password = ""
    if header.lower().startswith("basic "):
        try:
            password = base64.b64decode(header[6:]).decode("utf-8").partition(":")[2]
        except ValueError:
            password = ""
    if not secrets.compare_digest(password.encode(), APP_PASSWORD.encode()):
        return Response("Sign in to use Rexie.", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Rexie"'})
    return await call_next(request)

# RAG: /api/rag/...   Text-to-SQL: /api/sql/...   Dashboard: /api/usage/...   Issues + agent: /api/issues/...
app.include_router(rag_router)
app.include_router(sql_router)
app.include_router(dashboard_router)
app.include_router(issues_router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# serve the built React app (frontend/dist), if it has been built
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
