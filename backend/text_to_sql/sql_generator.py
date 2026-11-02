import re

from openai import OpenAI

from backend.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL_SQL


SYSTEM_PROMPT = (
    "You are an expert SQLite developer. "
    "Given the database schema and a question, write ONE SQLite SELECT query that answers it. "
    "Use only the tables and columns in the schema. "
    "Return only the SQL query, with no explanation."
)


class SQLGenerator:
    """
    Turns a question + database schema into a SQL query using an LLM.
    Uses Groq's OpenAI-compatible API (model set in backend/config.py).
    """

    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model_name = GROQ_MODEL_SQL
        self.client = None

    def generate(self, question: str, schema: str) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")

        # created on first use, so the server can start without a key
        if self.client is None:
            self.client = OpenAI(
                base_url=GROQ_BASE_URL,
                api_key=self.api_key,
            )

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
        return self._extract_sql(completion.choices[0].message.content)

    def _extract_sql(self, text: str) -> str:
        # models often wrap the query in ```sql ... ``` fences
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        return text.strip().rstrip(";").strip()
