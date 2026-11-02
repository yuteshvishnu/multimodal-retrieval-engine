from fastapi import APIRouter, Form
from backend.text_to_sql.schemas import QueryResponse

from backend.text_to_sql.pipeline import TextToSQLPipeline

router = APIRouter(prefix="/api/sql", tags=["Version-2: Text-to-SQL"])

# single pipeline instance for the whole app
pipeline = TextToSQLPipeline()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(query_text: str = Form("")):
    return pipeline.run(query_text)
