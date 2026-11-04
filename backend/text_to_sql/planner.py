"""
Scout Version-10: the steps. After the user confirms what the question means, the model
writes the plan as a short numbered list of plain-English steps (which data, how it's linked,
what's kept, how it's grouped, counted and ordered). The user checks each step; comments on
wrong ones re-run only this step (1 LLM call), never the meaning before it. The SQL is then
written to follow the confirmed steps, and the steps are shown with the answer.
"""
import json
import re
from typing import List, Optional

from backend.config import LLM_API_KEY, LLM_MODEL_SQL
from backend.llm import chat, make_client

MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You plan how to answer a question about a database, for a user who doesn't know SQL.\n"
    f"Write the plan as 3 to {MAX_STEPS} short numbered steps in plain English, in the order they "
    "happen: which data to start from, how it's linked to other data, which rows are kept or left "
    "out, how it's grouped, what's counted or added up, how it's sorted, and what's shown at the end. "
    "One action per step. Name the data in everyday words (orders, customers, stores, the order date, "
    "the employee); never use table or column names like order_date or employee_id, and no code formatting. "
    "Follow the meaning the user confirmed exactly; never add filters, columns or details it doesn't ask for.\n"
    # answer format (BIRD's convention): an evaluation compares the result's columns exactly
    "The last step shows only what the question asks for, as it's stored: no ids, extra counts or "
    "other extras, first and last names kept as two separate values, and numbers not rounded, unless "
    "the question asks for them. Never write 'optionally': decide.\n"
    "Never plan to change, add or delete data: only read it.\n"
    'Return only JSON: {"steps": ["step 1 text", "step 2 text", ...]}'
)


class StepPlanner:
    """Turns a confirmed meaning into steps the user can check (1 LLM call per try)."""

    def __init__(self):
        self.api_key = LLM_API_KEY
        self.model_name = LLM_MODEL_SQL
        self.client = None

    def steps(
        self, question: str, meaning: str, schema: str, context: str = "",
        history: List[dict] = (), tag: Optional[dict] = None,
    ) -> List[str]:
        """
        history: earlier tries the user corrected, each
            {"steps": [...], "comments": [{"step": 2, "comment": "..."}]}  (step numbers start at 1)
        """
        if not self.api_key:
            raise RuntimeError("The LLM API key is not set (LLM_PROVIDER and its key in .env).")
        if self.client is None:
            self.client = make_client(self.api_key)

        completion = chat(
            self.client,
            tag,
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.prompt(question, meaning, schema, context, history)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return self._parse(completion.choices[0].message.content or "")

    @staticmethod
    def prompt(question: str, meaning: str, schema: str, context: str, history: List[dict]) -> str:
        prompt = f"Schema:\n{schema}\n\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"Question: {question}\n\nWhat the user means (confirmed by the user):\n{meaning}"
        for i, past in enumerate(history, 1):
            shown = "\n".join(f"  {n}. {s}" for n, s in enumerate(past["steps"], 1))
            notes = "\n".join(f"  - step {c['step']}: {c['comment']}" for c in past.get("comments", []))
            prompt += (
                f"\n\nTry {i}: you wrote these steps:\n{shown}\n"
                f"The user said these steps are wrong:\n{notes}\n"
                "Fix those steps exactly as the user says. Keep the other steps unless the fix requires changing them."
            )
        return prompt

    @staticmethod
    def _parse(text: str) -> List[str]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group(0) if match else text)
        steps = [re.sub(r"^\s*\d+[.)]\s*", "", str(s)).strip() for s in data.get("steps", [])]
        steps = [s for s in steps if s][:MAX_STEPS]
        if not steps:
            raise ValueError("The model returned no steps.")
        return steps
