import re
import threading
from pathlib import Path
from typing import List, Optional

from backend.llm import chat, make_client

from backend.config import LLM_API_KEY, LLM_MODEL_SQL, SQL_REFRESHER


# Version-24: the highest or lowest value "per month", "per customer"... is the highest or lowest of the totals
GRANULARITY_RULE = (
    "When the question asks for the highest, lowest, maximum or minimum value at a granularity (monthly, "
    "yearly, per customer, per category...), first check whether several rows make up one unit of it. If "
    "they do, group by that unit, total the measure, and then take the highest or lowest of those totals. "
    "Take MAX or MIN of single rows only when each row already is one unit. "
)
# the same rule for the meaning and the plan, which use no SQL words
GRANULARITY_RULE_PLAIN = (
    "When the question asks for the highest or lowest value per month, year, customer, category or similar, "
    "and several records make up one of these, the value is the total for each one: the answer is the highest "
    "or lowest of those totals, not of single records. "
)

# Version-24: work out the exact metric first; count the things when the question compares how many there are
METRIC_RULE = (
    "Before choosing tables, joins and aggregate functions, decide which kind of measure is asked for: a count "
    "of entities, a count of transactions, a money amount, or another metric. Use the question, its hint and the "
    "schema's descriptions: a hint that defines a calculation takes precedence over ambiguous wording. "
    "when the question compares how many entities (customers, members, products...) belong to different "
    "categories, count them with conditional aggregation on the entities' own table, rather than joining "
    "transaction tables and adding up amounts, unless the question is about money spent or paid. Join other "
    "tables only when the metric needs them. "
)
METRIC_RULE_PLAIN = (
    "First work out which kind of measure is asked for: how many people or things, how many transactions, an "
    "amount of money, or something else, from the question, its hint and the data's descriptions. A hint that "
    "defines a calculation wins over unclear wording. When the question compares how many "
    "people or things belong to different groups, count them, without adding up money amounts or bringing in "
    "other records, unless the question is about money spent or paid. "
)
# Version-24: "which month", "per year": that part of the field, the same in the grouping and in the result
TIME_PART_RULE = (
    "When the question asks for a part of a field (the month, year or day of a date, or another part), take "
    "that part from the column and use the same expression in SELECT and GROUP BY: filter to the period asked, "
    "aggregate the measure, and return that part, not the full field, unless the full field is needed to tell "
    "results apart. "
)
TIME_PART_RULE_PLAIN = (
    "When the question asks for a part of a value (the month, the year or the day of a date, or another part), "
    "group by that part, keep only the period asked for, and show that part rather than the whole value, unless "
    "the whole value is needed to tell results apart. "
)

# Version-24: "average per event": divide by the entity asked for, not by a table that links to it
AVERAGE_RULE = (
    "When calculating an average per entity (per event, per customer...), divide by the number of that entity. "
    "Check whether the schema has a direct id for it or reaches it through an intermediate table, and don't treat "
    "an intermediate entity as the same as the target (budget lines are not events) unless the schema's keys or "
    "the data show they match one to one. "
)
AVERAGE_RULE_PLAIN = (
    "When the question asks for an average per person or thing, divide by the number of those people or things, "
    "not by the number of records that link to them, unless each of those records stands for exactly one. "
)

# Version-24: a record named by a value ("the customer who paid 634.8"): match it where the data holds it
VALUE_RULE = (
    "When the question identifies a record by a specific value (a date, a price, an amount), work out which column "
    "or expression holds that value before writing SQL, preferring a column described as holding it over a "
    "calculation, and make sure the condition picks out the intended record before computing anything from it. "
)
VALUE_RULE_PLAIN = (
    "When the question names a record by a specific value (a date, a price, an amount), say which stored value it "
    "matches, preferring a value the data stores directly over one worked out from others. "
)

# answer format (BIRD's convention): an evaluation compares the result's columns exactly
ANSWER_RULES = GRANULARITY_RULE + METRIC_RULE + TIME_PART_RULE + AVERAGE_RULE + VALUE_RULE + (
    "Return only the columns the question asks for, exactly as stored: no ids, extra counts or other "
    "extra columns, no joined text (keep first_name and last_name as separate columns), and no ROUND "
    "or other formatting, unless the question asks for it. "
    # Version-24: "respectively" lists the values wanted, in order
    "When the question lists items and says respectively (e.g. 'the totals for A, B and C respectively'), "
    "return one row with one value per item, as separate columns in the order listed, without labels. "
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
)

SYSTEM_PROMPT = (
    "You are an expert SQLite developer. "
    "Given the database schema and a question, write ONE SQLite SELECT query that answers it. "
    "Use only the tables and columns in the schema. "
    "Use the context for the current date and for what dates the data covers. "
    "If the user's confirmed meaning is given, follow it exactly: it decides anything the question leaves open. "
    + ANSWER_RULES +
    "Inside a text value, write an apostrophe twice: name = 'O''Brien'. "
    "Return only the SQL query, with no explanation."
)


FINAL_RULE = "Return only the SQL query, with no explanation."

# Version-23: what SQLite's commands do, with examples. Sent only with the calls that write SQL
REFRESHER = (Path(__file__).parent / "sqlite_refresher.md").read_text(encoding="utf-8").strip()


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
        self.refresher = SQL_REFRESHER

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
        instruction: str = "",
        examples: str = "",
        model: Optional[str] = None,
    ) -> str:
        """
        Version-22, a chain of the Full flow: instruction replaces the "only the SQL" rule (the chain's own
        way of working, e.g. notes first, then a ```sql block); examples are its worked examples; model
        overrides this Scout's model for this one call (the Mixed flow).
        """
        if not self.api_key:
            raise RuntimeError("The LLM API key is not set (LLM_PROVIDER and its key in .env).")

        budget.take()  # raises once the per-question limit is used up

        # created on first use, so the server can start without a key
        if self.client is None:
            self.client = make_client(self.api_key)

        prompt = f"Worked examples, on a different database:\n{examples}\n\n" if examples else ""
        prompt += f"Schema:\n{schema}\n\n"
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
        system = self._system_prompt()
        if instruction:
            system = system.replace(FINAL_RULE, instruction)
        if self.refresher:
            system += f"\n\n{REFRESHER}"

        completion = chat(
            self.client,
            tag,
            model=model or self.model_name,
            messages=[
                {"role": "system", "content": system},
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
