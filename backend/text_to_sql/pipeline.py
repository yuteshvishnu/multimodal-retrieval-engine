from typing import List

from backend.text_to_sql.database import SQLiteDatabase
from backend.text_to_sql.schema_index import SchemaIndex
from backend.text_to_sql.sql_generator import SQLGenerator

# how many tables the embedding search picks (bridge tables are added on top)
TOP_K_TABLES = 4


class TextToSQLPipeline:
    """
    Question -> pick relevant tables (embeddings) -> SQL (LLM) -> run on SQLite -> result table.

    use_retrieval=False sends the full schema instead (Version-2 behaviour), for comparison.
    """

    def __init__(self, use_retrieval: bool = True, top_k: int = TOP_K_TABLES):
        self.database = SQLiteDatabase()
        self.generator = SQLGenerator()
        self.top_k = top_k

        # tables are read once at startup; restart the server after adding .sqlite files
        self.tables = self.database.tables
        self.index = SchemaIndex(self.tables) if use_retrieval and self.tables else None

    def select_tables(self, query_text: str) -> List[dict]:
        if self.index is None:
            return self.tables
        return self.index.retrieve(query_text, self.top_k)

    def run(self, query_text: str) -> dict:
        if not query_text.strip():
            return {"error": "Please enter a question."}

        if not self.tables:
            return {"error": "No tables found. Add .sqlite files to the data folder and restart the server."}

        # 1) pick the relevant tables
        tables = self.select_tables(query_text)
        table_names = [table["name"] for table in tables]
        schema = "\n\n".join(table["schema"] for table in tables)
        print(f"[Pipeline] Using {len(tables)}/{len(self.tables)} tables: {table_names}")

        # 2) generate SQL
        try:
            sql = self.generator.generate(query_text, schema)
        except Exception as e:
            print(f"[Pipeline] SQL generation failed: {e}")
            return {"tables": table_names, "error": f"Could not generate SQL: {e}"}
        print(f"[Pipeline] Generated SQL: {sql}")

        # 3) run it
        try:
            columns, rows = self.database.run(sql)
        except Exception as e:
            print(f"[Pipeline] SQL execution failed: {e}")
            return {"sql": sql, "tables": table_names, "error": f"Query failed: {e}"}
        print(f"[Pipeline] Returned {len(rows)} rows")

        return {"sql": sql, "tables": table_names, "columns": columns, "rows": rows}
