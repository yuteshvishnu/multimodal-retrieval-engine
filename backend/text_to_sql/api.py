import json

from fastapi import APIRouter, Form
from backend.text_to_sql.schemas import InterpretResponse, QueryResponse

from backend.text_to_sql.pipeline import TextToSQLPipeline

router = APIRouter(prefix="/api/sql", tags=["Version-2: Text-to-SQL"])

# single pipeline instance for the whole app
pipeline = TextToSQLPipeline()


@router.post("/interpret", response_model=InterpretResponse)
def interpret_endpoint(query_text: str = Form(""), history: str = Form("[]")):
    """Version-5: up to 3 meanings of the question. history = JSON list of rejected rounds."""
    return pipeline.interpret(query_text, json.loads(history))


@router.post("/query", response_model=QueryResponse)
def query_endpoint(query_text: str = Form(""), meaning: str = Form("")):
    return pipeline.run(query_text, meaning)
