import re
import threading
from typing import List, Optional

from backend.llm import chat, make_client

from backend.config import LLM_API_KEY, LLM_MODEL_SQL


SYSTEM_PROMPT = (
    "You are an expert SQLite developer. "
    "Given the database schema and a question, write ONE SQLite SELECT query that answers it. "
    "Use only the tables and columns in the schema. "
    "Use the context for the current date and for what dates the data covers. "
    "If the user's confirmed meaning is given, follow it exactly: it decides anything the question leaves open. "
    # answer format (BIRD's convention): an evaluation compares the result's columns exactly
    "Return only the columns the question asks for, exactly as stored: no ids, extra counts or other "
    "extra columns, no joined text (keep first_name and last_name as separate columns), and no ROUND "
    "or other formatting, unless the question asks for it. "
    "When the question asks who, which or what, return only that, not the value used to rank or "
    "filter it (a total, a count), unless the question asks for it too. "
    "When the question asks yes/no and then asks for a value (Is it true…? If so, how many / how "
    "much / by what percent?), return only that value: it answers the yes/no by itself. Compute it "
    "in the direction the question states (the first thing minus the second, or as the hint's "
    "formula says) and keep it as computed, even if negative or 0: never replace it with 0, "
    "true/false or a message (no CASE WHEN … THEN 'true'). "
    "Leaving ids out applies only to what's shown: still group and join by the id (different records "
    "can share a name, e.g. two customers called Ana Kim), even when only names are shown. "
    "But when the question or its hint defines how a value is formed (e.g. 'the month is the 5th and "
    "6th characters of Date'), return that value in exactly that form, not the stored column. "
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
        self.api_key = LLM_API_KEY
        self.model_name = LLM_MODEL_SQL
        self.client = None
        # Version-9: returns the error book's one-line SQLite rules, added to the prompt
        self.rules = None

    def generate(
        self,
        question: str,
        schema: str,
        budget: CallBudget,
        context: str = "",
        feedback: str = "",
        meaning: str = "",
        temperature: float = 0,
        tag: Optional[dict] = None,
        steps: Optional[List[str]] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("The LLM API key is not set (LLM_PROVIDER and its key in .env).")

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
        if steps:
            prompt += "\n\nSteps confirmed by the user (the SQL must do exactly these):\n" + "\n".join(
                f"{n}. {step}" for n, step in enumerate(steps, 1)
            )
        if feedback:
            prompt += f"\n\nEarlier attempts had these problems; avoid them:\n{feedback}"

        completion = chat(
            self.client,
            tag,
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return self._extract_sql(completion.choices[0].message.content or "")

    def _system_prompt(self) -> str:
        rules = self.rules() if self.rules else []
        if not rules:
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + "\nSQLite rules (mistakes seen before, avoid them):\n" + "\n".join(f"- {r}" for r in rules)

    def _extract_sql(self, text: str) -> str:
        # models often wrap the query in ```sql ... ``` fences
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        return text.strip().rstrip(";").strip()
