"""
Text-to-SQL Version-5: understand the question before writing any SQL.

The model lists up to 3 things the question could mean, each with the choices that make it
different ("Cancelled orders: counted"). The user picks one, or says "none of these" and
explains; the next set of options uses that comment. The picked meaning is what the SQL
step follows.
"""
import json
import re
from typing import List, Optional

from backend.llm import chat, make_client

from backend.config import GROQ_API_KEY, GROQ_MODEL_SQL

MAX_OPTIONS = 3

SYSTEM_PROMPT = (
    "You help a user ask questions about a database. Before any SQL is written, work out what "
    "the question could mean.\n"
    f"List 1 to {MAX_OPTIONS} different meanings. Each meaning must differ from the others in at "
    "least one choice that would change the answer: which rows count, how ties are handled, which "
    "dates, what units, what to show. Use the schema and its column descriptions to spot these "
    "choices. Never list the same meaning twice in different words. If the question has only one "
    "sensible meaning, return just one.\n"
    "The FIRST meaning is always the plain, literal reading of the question. No meaning may add "
    "columns, details or filters the question doesn't ask for (if only names are asked for, don't "
    "add their city); meanings differ only in how they read what IS asked.\n"
    "Write for someone who doesn't know SQL or this database: plain everyday words only. Never use "
    "table or column names (no order_date, employee_id, order_items, unit_price_cents); say 'the order "
    "date', 'the employee', 'the price' instead. If a column has no clear everyday meaning, describe "
    "it as best you can. No SQL, no code formatting.\n"
    "If the question asks to change the data in any way (add, insert, update, edit, delete, remove, "
    "create, drop, rename, reset, increase...), describe it honestly as that change, never as a "
    "read-only question, and set changes_data to true.\n"
    "Return only JSON in this shape:\n"
    '{"options": [{"meaning": "one sentence saying what will be computed", '
    '"choices": ["short line per decision, e.g. Cancelled orders: counted"], "changes_data": false}]}'
)


class QuestionInterpreter:
    """Turns a question into up to 3 plain-language meanings for the user to pick from."""

    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model_name = GROQ_MODEL_SQL
        self.client = None

    def options(
        self, question: str, schema: str, context: str = "", history: List[dict] = (), tag: Optional[dict] = None,
        known: str = "",
    ) -> List[dict]:
        """
        history: earlier rounds the user rejected, each {"options": [...], "comment": "..."}.
        known: a saved meaning already shown as the first option (Version-12); the model then
        writes up to 2 others that differ from it.
        One LLM call. Returns [{"meaning": str, "choices": [str]}].
        """
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")

        # created on first use, so the server can start without a key
        if self.client is None:
            self.client = make_client(self.api_key)

        completion = chat(
            self.client,
            tag,
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.prompt(question, schema, context, history, known)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return self._parse(completion.choices[0].message.content or "", MAX_OPTIONS - 1 if known else MAX_OPTIONS)

    @staticmethod
    def prompt(question: str, schema: str, context: str, history: List[dict], known: str = "") -> str:
        prompt = f"Schema:\n{schema}\n\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"Question: {question}"
        for i, past in enumerate(history, 1):
            shown = "\n".join(f"  - {o['meaning']}" for o in past["options"])
            prompt += (
                f"\n\nRound {i}: you offered these meanings:\n{shown}\n"
                f"The user said none of them is right and explained: \"{past['comment']}\"\n"
                "Use the user's explanation exactly; it decides anything it mentions."
            )
        if known:
            prompt += (
                f"\n\nThis meaning is already offered to the user as the first option:\n{known}\n"
                f"List 1 to {MAX_OPTIONS - 1} OTHER meanings that differ from it in at least one choice. "
                "Never repeat it in other words."
            )
        return prompt

    @staticmethod
    def _parse(text: str, limit: int = MAX_OPTIONS) -> List[dict]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group(0) if match else text)
        options = []
        for item in data.get("options", [])[:limit]:
            meaning = str(item.get("meaning", "")).strip()
            if meaning and meaning not in (o["meaning"] for o in options):
                options.append({
                    "meaning": meaning,
                    "choices": [str(c).strip() for c in item.get("choices", [])],
                    "changes_data": item.get("changes_data") is True,
                })
        if not options:
            raise ValueError("The model returned no meanings.")
        return options
