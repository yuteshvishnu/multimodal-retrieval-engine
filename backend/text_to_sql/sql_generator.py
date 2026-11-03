import re
import threading

from backend.llm import make_client

from backend.config import GROQ_API_KEY, GROQ_MODEL_SQL


SYSTEM_PROMPT = (
    "You are an expert SQLite developer. "
    "Given the database schema and a question, write ONE SQLite SELECT query that answers it. "
    "Use only the tables and columns in the schema. "
    "Use the context for the current date and for what dates the data covers. "
    "If the user's confirmed meaning is given, follow it exactly: it decides anything the question leaves open. "
    "Return only the SQL query, with no explanation."
)


class CallLimitReached(RuntimeError):
    pass


class CallBudget:
    """
    Hard stop on LLM calls for one question. Every call must take a slot first;
    once the limit is used up, no further call can be made.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise CallLimitReached(f"LLM call limit of {self.limit} reached")
            self.used += 1

    @property
    def left(self) -> int:
        return self.limit - self.used


class SQLGenerator:
    """
    Turns a question + database schema into a SQL query using an LLM.
    Uses Groq's OpenAI-compatible API (model set in backend/config.py).
    """

    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model_name = GROQ_MODEL_SQL
        self.client = None

    def generate(
        self,
        question: str,
        schema: str,
        budget: CallBudget,
        context: str = "",
        feedback: str = "",
        meaning: str = "",
        temperature: float = 0,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")

        budget.take()  # raises once the per-question limit is used up

        # created on first use, so the server can start without a key
        if self.client is None:
            self.client = make_client(self.api_key)

        prompt = f"Schema:\n{schema}\n\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"Question: {question}"
        if meaning:
            prompt += f"\n\nWhat the user means (confirmed by the user):\n{meaning}"
        if feedback:
            prompt += f"\n\nEarlier attempts had these problems; avoid them:\n{feedback}"

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return self._extract_sql(completion.choices[0].message.content or "")

    def _extract_sql(self, text: str) -> str:
        # models often wrap the query in ```sql ... ``` fences
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        return text.strip().rstrip(";").strip()
