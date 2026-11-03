import json
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from backend.text_to_sql.catalog import PAGE_SIZE, check_new_tables, list_tables, save_upload
from backend.text_to_sql.schemas import InterpretResponse, QueryResponse
from backend.text_to_sql.table_summaries import TableSummarizer, save_summary

from backend.text_to_sql.pipeline import TextToSQLPipeline

router = APIRouter(prefix="/api/sql", tags=["Text-to-SQL"])

# single pipeline instance for the whole app
pipeline = TextToSQLPipeline()

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@router.post("/interpret", response_model=InterpretResponse)
def interpret_endpoint(query_text: str = Form(""), history: str = Form("[]"), previous: str = Form("")):
    """
    Version-5: up to 3 meanings of the question. history = JSON list of rejected rounds.
    previous = JSON {question, meaning, sql} of the answer this follows up on (optional).
    """
    return pipeline.interpret(query_text, json.loads(history), _previous(previous))


@router.post("/query", response_model=QueryResponse)
def query_endpoint(query_text: str = Form(""), meaning: str = Form(""), previous: str = Form("")):
    return pipeline.run(query_text, meaning, _previous(previous))


def _previous(text: str):
    """The earlier answer a follow-up refers to, or None."""
    try:
        data = json.loads(text) if text else None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


@router.get("/results/{result_id}")
def result_page_endpoint(result_id: str, page: int = 1):
    """Version-8: one page (10 rows) of a saved result, without running the query again."""
    return pipeline.result_page(result_id, page)


# ---------- Version-6: the Tables view ----------

def _table(database: str, name: str) -> dict:
    table = pipeline.find_table(database, name)
    if table is None:
        raise HTTPException(404, f"No table '{name}' in database '{database}'.")
    return table


@router.get("/tables")
def tables_endpoint(page: int = 1, search: str = ""):
    """Every table Rexie knows, 10 per page, with what Rexie understands about it."""
    return list_tables(pipeline.tables, page, search)


@router.get("/tables/{database}/{name}/rows")
def rows_endpoint(database: str, name: str, page: int = 1):
    """One page (10 rows) of a table's data."""
    table = _table(database, name)
    pages = max(1, -(-table["row_count"] // PAGE_SIZE))
    page = min(max(1, page), pages)
    columns, rows = pipeline.database.preview(database, name, page, PAGE_SIZE)
    return {"columns": columns, "rows": rows, "page": page, "pages": pages, "total": table["row_count"]}


@router.post("/tables/{database}/{name}/rewrite")
def rewrite_endpoint(database: str, name: str, current: str = Form(...), feedback: str = Form("[]")):
    """
    A new description that follows the user's corrections (1 LLM call). Not saved:
    the user reviews it and either saves it or asks for another try.
    """
    table = _table(database, name)
    corrections = [c.strip() for c in json.loads(feedback) if c.strip()]
    if not corrections:
        raise HTTPException(400, "Say what is wrong with the description.")
    try:
        summary, problems = TableSummarizer(pipeline.database).rewrite(
            table, [t["name"] for t in pipeline.tables], current, corrections
        )
    except Exception as e:
        raise HTTPException(502, f"Could not rewrite the description: {e}")
    return {"description": summary, "problems": problems, "llm_calls": 1}


@router.post("/tables/{database}/{name}/save")
def save_endpoint(database: str, name: str, description: str = Form(...), feedback: str = Form("[]")):
    """Saves the description the user confirmed, then re-embeds it for table search."""
    table = _table(database, name)
    save_summary(table, " ".join(description.split()), [c for c in json.loads(feedback) if c.strip()])
    pipeline.reload()
    return {"table": list_tables([pipeline.find_table(database, name)], 1)["items"][0]}


@router.delete("/databases/{database}")
def delete_database_endpoint(database: str):
    """
    Removes a database: its folder (the .sqlite file, descriptions, summaries, embeddings),
    then reloads. Only a database in its own folder inside the data folder can be removed.
    """
    folder = next((Path(t["folder"]) for t in pipeline.tables if t["database"] == database), None)
    if folder is None:
        raise HTTPException(404, f"No database called '{database}'.")
    data_dir, folder = pipeline.database.data_dir.resolve(), folder.resolve()
    if folder == data_dir or not folder.is_relative_to(data_dir) or len(list(folder.glob("*.sqlite"))) != 1:
        raise HTTPException(400, "This database shares its folder with other files, so it can't be removed here.")
    shutil.rmtree(folder)
    pipeline.reload()
    return {"deleted": database}


@router.post("/tables/upload")
def upload_endpoint(file: UploadFile = File(...)):
    """
    Adds a .sqlite database: saves it, loads its tables, writes an LLM summary for each
    (1 call per table, up to 3 if a summary breaks the rules) and embeds them. No restart needed.
    """
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is larger than 200 MB.")
    try:
        path = save_upload(pipeline.database.data_dir, file.filename, content, pipeline.tables)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        names = check_new_tables(path, pipeline.tables)
    except ValueError as e:
        shutil.rmtree(path.parent, ignore_errors=True)
        raise HTTPException(400, str(e))

    pipeline.reload()
    warning = None
    try:
        TableSummarizer(pipeline.database).build(folders=[path.parent])
        pipeline.reload()
    except Exception as e:
        warning = f"Added, but descriptions could not be written yet ({e}); they are built from column names."
    return {"database": path.stem, "tables": names, "warning": warning}
