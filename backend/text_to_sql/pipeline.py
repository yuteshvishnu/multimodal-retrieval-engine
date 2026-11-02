from backend.text_to_sql.database import SQLiteDatabase
from backend.text_to_sql.sql_generator import SQLGenerator


class TextToSQLPipeline:
    """
    Question -> SQL (LLM) -> run on SQLite -> result table.
    """

    def __init__(self):
        self.database = SQLiteDatabase()
        self.generator = SQLGenerator()

        # schema is read once at startup; restart the server after adding .sqlite files
        self.schema = self.database.schema()

    def run(self, query_text: str) -> dict:
        if not query_text.strip():
            return {"error": "Please enter a question."}

        if not self.schema:
            return {"error": "No tables found. Add .sqlite files to the data folder and restart the server."}

        # 1) generate SQL
        try:
            sql = self.generator.generate(query_text, self.schema)
        except Exception as e:
            print(f"[Pipeline] SQL generation failed: {e}")
            return {"error": f"Could not generate SQL: {e}"}
        print(f"[Pipeline] Generated SQL: {sql}")

        # 2) run it
        try:
            columns, rows = self.database.run(sql)
        except Exception as e:
            print(f"[Pipeline] SQL execution failed: {e}")
            return {"sql": sql, "error": f"Query failed: {e}"}
        print(f"[Pipeline] Returned {len(rows)} rows")

        return {"sql": sql, "columns": columns, "rows": rows}
