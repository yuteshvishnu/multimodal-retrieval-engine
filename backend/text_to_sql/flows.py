"""
Version-22: the Full flow, after "Ask Me Anything: A simple strategy for prompting language
models" (Arora, Narayan, ... Frederic Sala, Christopher Re), and what can be replayed from it.

Scout's original flow writes 3 SQL tries from one prompt and takes the majority. The paper's idea:
ask the SAME task in DIFFERENT ways, through several imperfect prompt chains, and combine the
answers. As in the paper, each chain has two calls and its own worked examples:

    question()   rewrites the question in the chain's own style (the same meaning: never a
                 different reading, which would make an answer wrong on purpose)
    answer()     writes the SQL in the chain's own way

    chain     meaning written as                  SQL written by first
    direct    simple: plain words                 nothing: straight to the SQL
    explore   data: exact tables, columns, values noting what to use
    plan      answer: which values, how computed  planning the steps
    shape     hint: starting from the hint        describing one row of the result

Every chain's SQL goes through Scout's own checks (safety, valid SQL, values in the data, grouping
by a shared name) and is run; a chain that fails abstains. No retry. 8 LLM calls.

Answers: chains whose rows are the same (as a set, as BIRD compares them) give one answer. The
live answer is a majority vote: the paper's own choice for open-ended answers ("using WS is complex
with the open-ended output space"). Grouping by rows is what makes weak supervision possible here,
so it's replayed afterwards from the saved chains, with no LLM calls:

    majority_vote(chains)   the first 3 or all 4 chains
    weighted_vote(chains)   weak supervision: each chain's vote weighs log(acc / (1 - acc)), with acc
                            learned from how often the chains agree, across all the questions and
                            after all the answers are in (as the paper does), never from correct
                            answers. If mistakes are independent, agree(i, j) ~ acc(i) x acc(j), so
                            acc(i) = sqrt(agree(i,j) x agree(i,k) / agree(j,k)), the "triplet" method
                            (Fu, Chen, Sala et al., "Fast and Three-rious", 2020). Unlike the paper
                            (Varma, Sala et al., 2019), it doesn't learn which chains copy each other.
    adaptive_vote(chains)   our idea, not the paper's: the first 2 chains agree on an answer that has
                            data -> done (4 calls); otherwise a majority of all 4 (8 calls)

Mixed Model (our extension, not the paper's): the 3 voters are 3 different models, from 3 labs, because
changing only the wording leaves mistakes that come from sharing one model. Each model answers every
question once. For each question, 3 of the 4 rewrite styles and 3 of the 4 ways of writing SQL are shuffled
among the models, so no model is tied to a style, and the shuffle comes from the question, so it's repeatable. Its
live answer is a majority vote; its weak-supervision answer is worked out afterwards, per model.

Adaptive + English program (Version-24, our idea): the same Adaptive flow, but the model never writes SQL. It
writes one sentence in the strict format of backend/english_to_sql/FORMAT.md, and the program english2sqlite.py
turns that sentence into SQL, with no AI. A sentence the program rejects gets the one retry, told why. The
format's `raw` shortcuts are blocked, because they would let the model write SQL after all.

Adaptive also runs live, without Full: the first 2 chains, then the other 2 only if needed. Live, a
chain whose SQL the checks reject gets one more try, told why (Full stays as in the paper: no retry,
so a replayed Adaptive has none either). Two chains agreeing on no rows isn't enough to stop:
agreeing on nothing is weak evidence, and the question usually assumes there's something to find.
"""
import hashlib
import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import MIXED_MODELS, SQL_TEMPERATURE
from backend.english_to_sql.english2sqlite import compile_english
from backend.llm import chat, make_client
from backend.text_to_sql import semantic_check
from backend.text_to_sql.semantic_check import default_spec, parse_spec, spec_lines, with_question
from backend.text_to_sql.sql_generator import (
    ANSWER_RULES, AVERAGE_RULE_PLAIN, VALUE_RULE_PLAIN, GRANULARITY_RULE_PLAIN, METRIC_RULE_PLAIN, TIME_PART_RULE_PLAIN, CallBudget,
    CallLimitReached,
)

FULL_CALLS = 8
ADAPTIVE_MAX_CALLS = 12          # all 4 chains, each with one retry
ADAPTIVE_CHAINS = 2
MIN_QUESTIONS_FOR_WEIGHTS = 10   # fewer questions with all chains answering: equal weights
ACCURACY_RANGE = (0.05, 0.95)    # estimates are clipped: a chain is never certain or worthless

FORMAT_RULE = "The meaning decides what to compute; the format rules above decide which columns to return, and how."
SQL_ONLY = f"{FORMAT_RULE} Return only the SQL query, with no explanation."
THEN_SQL = f"{FORMAT_RULE} Then write the query in a ```sql code block. Nothing after the block."

# The worked examples are about the practice retail store, never a benchmark's questions.
CHAINS = [
    {
        "id": "direct", "style": "simple",
        "style_prompt": "Write it in plain everyday words, as you'd explain it to someone who doesn't know the database.",
        "instruction": SQL_ONLY,
        "meaning_examples": (
            "Question: How many customers signed up in 2021?\n"
            '{"meaning": "Count the customers whose signup date is in 2021.", "measure": "entity count", "grain": "none", "extreme": "none", "date_part": "none", "values_per_row": 0}\n\n'
            "Question: Which store has the most employees?\n"
            '{"meaning": "Find the store with the highest number of employees, and give its name.", "measure": "other", "grain": "store", "extreme": "highest", "date_part": "none", "values_per_row": 0}'
        ),
        "sql_examples": (
            "Question: How many customers signed up in 2021?\n"
            "SELECT COUNT(*) FROM customers WHERE substr(signup_date, 1, 4) = '2021'\n\n"
            "Question: Which store has the most employees?\n"
            "SELECT s.store_name FROM stores s JOIN employees e ON e.store_id = s.store_id "
            "GROUP BY s.store_id ORDER BY COUNT(*) DESC LIMIT 1"
        ),
    },
    {
        "id": "explore", "style": "data",
        "style_prompt": ("Write it in terms of the data: the exact tables, columns and stored values (spelled as in the "
                         "schema and the values listed) that each part of the question uses, and how the tables link."),
        "instruction": ("Before writing SQL, answer in a few short lines under 'Notes:': which table holds each thing "
                        "the question asks about; which column, and which stored values, spelled as in the schema; how "
                        f"those tables link; which rows are kept. {THEN_SQL}"),
        "meaning_examples": (
            "Question: What is the total refunded amount?\n"
            '{"meaning": "Add up payments.amount_cents over the rows where payments.status = \'refunded\'.", "measure": "money", "grain": "none", "extreme": "none", "date_part": "none", "values_per_row": 0}\n\n'
            "Question: Which suppliers are based in Canada?\n"
            '{"meaning": "Take suppliers.supplier_name over the rows where suppliers.country = \'Canada\'.", "measure": "other", "grain": "none", "extreme": "none", "date_part": "none", "values_per_row": 0}'
        ),
        "sql_examples": (
            "Question: What is the total refunded amount?\n"
            "Notes: payments holds the amounts, in amount_cents; refunded is status = 'refunded'; one table.\n"
            "```sql\nSELECT SUM(amount_cents) FROM payments WHERE status = 'refunded'\n```\n\n"
            "Question: Which suppliers are based in Canada?\n"
            "Notes: suppliers holds supplier_name and country; the stored value is 'Canada'; one table.\n"
            "```sql\nSELECT supplier_name FROM suppliers WHERE country = 'Canada'\n```"
        ),
    },
    {
        "id": "plan", "style": "answer",
        "style_prompt": ("Say which values the question asks for, only those, and exactly how each is computed: which "
                         "rows count, and what is added, counted or averaged."),
        "instruction": ("Before writing SQL, list the plan under 'Steps:' in a few short numbered steps: which data to "
                        f"start from, how it links, which rows are kept, how it's grouped and computed. {THEN_SQL}"),
        "meaning_examples": (
            "Question: Who placed the most orders?\n"
            '{"meaning": "The customer\'s first name and last name. Count each customer\'s orders, and keep the customer '
            'with the highest count.", "measure": "other", "grain": "customer", "extreme": "highest", "date_part": "none", "values_per_row": 0}\n\n'
            "Question: What is the average salary of cashiers?\n"
            '{"meaning": "The average of the salaries of the employees whose title is Cashier.", "measure": "money", "grain": "none", "extreme": "none", "date_part": "none", "values_per_row": 0}'
        ),
        "sql_examples": (
            "Question: Who placed the most orders?\n"
            "Steps:\n1. Start from the orders.\n2. Link each to its customer.\n3. Group by customer and count.\n"
            "4. Keep the highest count, and show the names.\n"
            "```sql\nSELECT c.first_name, c.last_name FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.customer_id ORDER BY COUNT(*) DESC LIMIT 1\n```\n\n"
            "Question: What is the average salary of cashiers?\n"
            "Steps:\n1. Start from the employees.\n2. Keep the cashiers.\n3. Average their salaries.\n"
            "```sql\nSELECT AVG(salary) FROM employees WHERE title = 'Cashier'\n```"
        ),
    },
    {
        "id": "shape", "style": "hint",
        "style_prompt": ("Start from the definitions in the hint, if there is one, and say what each one means for this "
                         "question; then say what is computed from them."),
        "instruction": ("Before writing SQL, say under 'Result:' what one row of the result is: its values, in order, "
                        f"and nothing else. {THEN_SQL}"),
        "meaning_examples": (
            "Question: How many large orders are there? (Hint: a large order has an item with quantity over 5)\n"
            '{"meaning": "By the hint, an order is large if any of its items has a quantity over 5. Count such orders, '
            'each once.", "measure": "transaction count", "grain": "none", "extreme": "none", "date_part": "none", "values_per_row": 0}\n\n'
            "Question: Which products are premium? (Hint: premium means a unit price over 10000 cents)\n"
            '{"meaning": "By the hint, a product is premium if its unit price is over 10000 cents. Give the names of '
            'those products.", "measure": "other", "grain": "none", "extreme": "none", "date_part": "none", "values_per_row": 0}'
        ),
        "sql_examples": (
            "Question: How many large orders are there? (Hint: a large order has an item with quantity over 5)\n"
            "Result: one row, one value: the number of large orders.\n"
            "```sql\nSELECT COUNT(DISTINCT order_id) FROM order_items WHERE quantity > 5\n```\n\n"
            "Question: Which products are premium? (Hint: premium means a unit price over 10000 cents)\n"
            "Result: one row per product, one value: its name.\n"
            "```sql\nSELECT product_name FROM products WHERE unit_price_cents > 10000\n```"
        ),
    },
]
CHAIN_IDS = [c["id"] for c in CHAINS]

MEANING_SYSTEM = (
    "You restate a question about a database so that SQL can be written from it. Restate exactly what the question "
    "asks: add no assumptions, filters or details it doesn't ask for, and follow the hint if one is given. When the "
    "question names a value, use how the data stores it (from the schema's listed values). Say only what to find and "
    "compute, never how to show it: don't ask to combine or reformat values, add ids or other extra values, or round. "
    + (GRANULARITY_RULE_PLAIN + METRIC_RULE_PLAIN + TIME_PART_RULE_PLAIN + AVERAGE_RULE_PLAIN + VALUE_RULE_PLAIN
       ).replace("{", "{{").replace("}", "}}") +
    "Don't write SQL. {style}\n"
    "Also name the measure: what kind of number the answer is. \"entity count\": how many people or things "
    "(customers, members, products). \"transaction count\": how many events or records (orders, transactions, "
    "visits). \"money\": an amount paid, spent, earned or charged. \"other\": anything else, including names, "
    "dates, percentages, ratios, averages that aren't money, and quantities.\n"
    "Then say what the answer must be: \"grain\": the unit the question totals or ranks by (month, year, customer, "
    "product...), or \"none\"; \"extreme\": \"highest\", \"lowest\" or \"none\"; \"date_part\": \"year\" or "
    "\"month\" when the answer is that part of a date, else \"none\"; \"values_per_row\": how many values each "
    "result row holds when the question fixes it (e.g. 3 for 'for A, B and C respectively'), else 0.\n"
    'Return only JSON: {{"meaning": "one to four sentences", "measure": "entity count, transaction count, money '
    'or other", "grain": "...", "extreme": "...", "date_part": "...", "values_per_row": 0}}\n\n'
    "Worked examples, on a different database:\n{examples}"
)


SEMANTIC_CACHE: Dict = {}  # which tables have several rows per unit: the data doesn't change during a run
ENGLISH_FLOW = "adaptive_english"
ADAPTIVE_FLOWS = ("adaptive", ENGLISH_FLOW)


def format_guide() -> str:
    """The query parts of FORMAT.md, without the raw shortcut: only what a read query can use."""
    text = (Path(__file__).parent.parent / "english_to_sql" / "FORMAT.md").read_text(encoding="utf-8")
    body = text[text.index("## Basic rules"):text.index("## Changing data")]
    return "\n".join(line for line in body.splitlines() if "`raw " not in line).strip()


FORMAT_GUIDE = format_guide()
SENTENCE_ONLY = f"{FORMAT_RULE} Return only the sentence, with no explanation."
THEN_SENTENCE = f"{FORMAT_RULE} Then write the sentence in a ```text code block. Nothing after the block."

# The same 4 ways of working and the same worked examples as CHAINS, written as sentences instead of SQL.
ENGLISH_WRITING = {
    "direct": {
        "instruction": SENTENCE_ONLY,
        "examples": (
            "Question: How many customers signed up in 2021?\n"
            'how many customers where signup_date starts with "2021"\n\n'
            "Question: Which store has the most employees?\n"
            "get s.store_name from stores as s join employees as e on e.store_id = s.store_id "
            "group by s.store_id sorted by count of rows descending limit 1"
        ),
    },
    "explore": {
        "instruction": ("Before writing the sentence, answer in a few short lines under 'Notes:': which table holds each "
                        "thing the question asks about; which column, and which stored values, spelled as in the "
                        f"schema; how those tables link; which rows are kept. {THEN_SENTENCE}"),
        "examples": (
            "Question: What is the total refunded amount?\n"
            "Notes: payments holds the amounts, in amount_cents; refunded is status = 'refunded'; one table.\n"
            '```text\nget sum of amount_cents from payments where status is "refunded"\n```\n\n'
            "Question: Which suppliers are based in Canada?\n"
            "Notes: suppliers holds supplier_name and country; the stored value is 'Canada'; one table.\n"
            '```text\nget supplier_name from suppliers where country is "Canada"\n```'
        ),
    },
    "plan": {
        "instruction": ("Before writing the sentence, list the plan under 'Steps:' in a few short numbered steps: which "
                        f"data to start from, how it links, which rows are kept, how it's grouped and computed. {THEN_SENTENCE}"),
        "examples": (
            "Question: Who placed the most orders?\n"
            "Steps:\n1. Start from the orders.\n2. Link each to its customer.\n3. Group by customer and count.\n"
            "4. Keep the highest count, and show the names.\n"
            "```text\nget c.first_name, c.last_name from orders as o join customers as c on o.customer_id = c.customer_id "
            "group by c.customer_id sorted by count of rows descending limit 1\n```\n\n"
            "Question: What is the average salary of cashiers?\n"
            "Steps:\n1. Start from the employees.\n2. Keep the cashiers.\n3. Average their salaries.\n"
            '```text\nget average of salary from employees where title is "Cashier"\n```'
        ),
    },
    "shape": {
        "instruction": ("Before writing the sentence, say under 'Result:' what one row of the result is: its values, in "
                        f"order, and nothing else. {THEN_SENTENCE}"),
        "examples": (
            "Question: How many large orders are there? (Hint: a large order has an item with quantity over 5)\n"
            "Result: one row, one value: the number of large orders.\n"
            "```text\nget count of unique order_id from order_items where quantity is over 5\n```\n\n"
            "Question: Which products are premium? (Hint: premium means a unit price over 10000 cents)\n"
            "Result: one row per product, one value: its name.\n"
            "```text\nget product_name from products where unit_price_cents is over 10000\n```"
        ),
    },
}
ENGLISH_CHAINS = [{**c, "english": True, **ENGLISH_WRITING[c["id"]]} for c in CHAINS]

ENGLISH_SYSTEM = (
    "You write one database query as one sentence in a strict English format. A program turns your sentence into "
    "SQLite, exactly as written, and rejects anything that doesn't fit the format. Never write SQL. "
    "Use only the tables and columns in the schema. "
    "Use the context for the current date and for what dates the data covers. "
    "If the user's confirmed meaning is given, follow it exactly: it decides anything the question leaves open. "
    + ANSWER_RULES +
    "\n\nThe format:\n" + FORMAT_GUIDE
)


class SentenceRejected(ValueError):
    """The program couldn't turn the sentence into one read query."""


def sentence_to_sql(sentence: str) -> str:
    """The model's sentence becomes SQL here, by the program and not by a model: one read query, or rejected."""
    text = " ".join(line.strip() for line in sentence.splitlines() if line.strip()).rstrip(" ;")
    if not text:
        raise SentenceRejected("The sentence is empty.")
    outside_quotes = re.sub(r"""\"(?:[^"]|"")*\"|'(?:[^']|'')*'|`(?:[^`]|``)+`""", " ", text)
    if text.startswith(".") or re.search(r"\braw\b", outside_quotes, re.IGNORECASE):
        raise SentenceRejected("Raw SQL and shell commands aren't allowed: write it in the format.")
    try:
        commands = compile_english(text)
    except Exception as e:
        raise SentenceRejected(str(e)) from e
    if len(commands) != 1 or not re.match(r"\s*(SELECT|WITH)\b", commands[0], re.IGNORECASE):
        raise SentenceRejected("Write exactly one query that only reads data.")
    return commands[0].strip().rstrip(";").strip()


def mixed_chains(question: str) -> List[dict]:
    """
    One chain per model, in a shuffled order (ties go to the earliest, so no model gets an edge), each with
    a different rewrite style and SQL method. The same question always gets the same shuffle.
    """
    rng = random.Random(int(hashlib.sha256(question.encode()).hexdigest()[:16], 16))
    styles, methods = rng.sample(CHAINS, len(MIXED_MODELS)), rng.sample(CHAINS, len(MIXED_MODELS))
    chains = [{
        "id": model["voter"], "model": model["id"], "style": style["style"], "method": method["id"],
        "style_prompt": style["style_prompt"], "meaning_examples": style["meaning_examples"],
        "instruction": method["instruction"], "sql_examples": method["sql_examples"],
    } for model, style, method in zip(MIXED_MODELS, styles, methods)]
    rng.shuffle(chains)
    return chains


def answer_key(rows: List[list]) -> str:
    """The same rows, in any order and with any repeats, give the same key. For agreement between chains,
    decimals are rounded so tiny float differences don't split them; only scoring is exact."""
    unique = {repr(tuple(round(v, 4) if isinstance(v, float) else v for v in row)) for row in rows}
    return hashlib.sha256("\n".join(sorted(unique)).encode()).hexdigest()[:16]


# ---------- the votes: plain functions, so saved chains can be replayed with no LLM calls ----------

def confidence_of(answered: int, agreeing: int) -> Optional[str]:
    """The same rule in every flow. high: every chain that answered agrees (at least 2 answered)."""
    if not agreeing:
        return None
    if agreeing >= 2 and agreeing == answered:
        return "high"
    return "medium" if agreeing >= 2 else "low"


def _pick(chains: List[dict], weights: Dict[str, float]) -> Optional[dict]:
    """
    The highest-scoring answer; a tie goes to the earliest chain. Empty answers only win when no chain
    found data (Version-24): a question usually assumes there's something to find. They still count as
    disagreeing, so the confidence drops.
    """
    ok = [c for c in chains if c.get("status") == "ok"]
    if not ok:
        return None
    voters = [c for c in ok if not is_empty(c.get("rows"))] or ok
    voters = [c for c in voters if not c.get("flags")] or voters  # Version-24: answers that pass the checks first
    scores: Dict[str, float] = {}
    for chain in voters:
        scores[chain["answer"]] = scores.get(chain["answer"], 0.0) + weights.get(chain["chain"], 1.0)
    best = max(voters, key=lambda c: (round(scores[c["answer"]], 6), -voters.index(c)))
    group = [c for c in voters if c["answer"] == best["answer"]]
    return {"winner": best, "agreeing": [c["chain"] for c in group], "confidence": confidence_of(len(ok), len(group))}


def is_empty(rows: Optional[List[list]]) -> bool:
    """No rows, or only NULLs. Unknown (None, not saved) doesn't count as empty."""
    return rows is not None and all(value is None for row in rows for value in row)


def should_stop_early(chains: List[dict]) -> bool:
    """Adaptive: both of the first chains answered, with the same rows, those rows hold data, and neither is
    flagged by the checks (Version-24: two chains making the same mistake isn't evidence)."""
    decided = majority_vote(chains[:ADAPTIVE_CHAINS])
    flagged = any(c.get("flags") for c in chains[:ADAPTIVE_CHAINS])
    return bool(decided and len(decided["agreeing"]) == ADAPTIVE_CHAINS and not flagged
                and not is_empty(decided["winner"].get("rows")))


def majority_vote(chains: List[dict], first: Optional[int] = None) -> Optional[dict]:
    return _pick(chains[:first] if first else chains, {})


def fit_accuracies(questions: List[List[dict]]) -> Optional[Dict[str, float]]:
    """
    Each chain's estimated accuracy, from how often the chains agree over these questions (a chain that
    abstained doesn't count). Triplet method, averaged over the triplets each chain is in.
    None until enough questions have every chain answering.
    """
    pairs: Dict[frozenset, List[int]] = {}
    complete = 0
    voter_count = max((len(chains) for chains in questions), default=0)
    for chains in questions:
        ok = [c for c in chains if c.get("status") == "ok"]
        complete += len(ok) == voter_count
        for x in range(len(ok)):
            for y in range(x + 1, len(ok)):
                count = pairs.setdefault(frozenset((ok[x]["chain"], ok[y]["chain"])), [0, 0])
                count[0] += ok[x]["answer"] == ok[y]["answer"]
                count[1] += 1
    if complete < MIN_QUESTIONS_FOR_WEIGHTS:
        return None
    rate = {key: max(agreed / both, 1e-3) for key, (agreed, both) in pairs.items() if both}
    out = {}
    voters = sorted({v for key in pairs for v in key})
    for i in voters:
        others = [c for c in voters if c != i]
        estimates = []
        for x in range(len(others)):
            for y in range(x + 1, len(others)):
                j, k = others[x], others[y]
                try:
                    estimates.append(math.sqrt(rate[frozenset((i, j))] * rate[frozenset((i, k))] / rate[frozenset((j, k))]))
                except KeyError:
                    continue
        if not estimates:
            return None
        out[i] = min(max(sum(estimates) / len(estimates), ACCURACY_RANGE[0]), ACCURACY_RANGE[1])
    return out


def weighted_vote(chains: List[dict], accuracies: Optional[Dict[str, float]]) -> Optional[dict]:
    weights = {c: math.log(a / (1 - a)) for c, a in (accuracies or {}).items()}
    return _pick(chains, weights)


def adaptive_vote(chains: List[dict]) -> Optional[dict]:
    """Our idea, not the paper's: what stopping early would have given, and how many calls it would have used."""
    if should_stop_early(chains):
        return {**majority_vote(chains[:ADAPTIVE_CHAINS]), "calls": ADAPTIVE_CHAINS * 2}
    decided = majority_vote(chains)
    return {**decided, "calls": FULL_CALLS} if decided else None


# ---------- the live run ----------

class FlowRunner:
    """Runs one question through the Full or Adaptive flow, using a Scout pipeline's own parts."""

    def __init__(self, pipeline):
        self.p = pipeline
        self._client = None

    def meaning(self, chain: dict, question: str, schema: str, context: str, budget: CallBudget, tag: dict) -> tuple:
        """question(): the chain's restatement, and a spec of what the answer must be (Version-24)."""
        budget.take()
        if self._client is None:
            self._client = make_client(self.p.generator.api_key)
        completion = chat(
            self._client, {**tag, "stage": "meaning", "chain": chain["id"]},
            model=chain.get("model") or self.p.generator.model_name,
            messages=[
                {"role": "system", "content": MEANING_SYSTEM.format(style=chain["style_prompt"],
                                                                    examples=chain["meaning_examples"])},
                {"role": "user", "content": f"Schema:\n{schema}\n\nContext:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        text = completion.choices[0].message.content or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            parsed = json.loads(match.group(0) if match else text)
            return str(parsed.get("meaning") or "").strip(), parse_spec(parsed)
        except (json.JSONDecodeError, AttributeError):
            return text.strip(), default_spec()

    def sentence(self, chain: dict, question: str, schema: str, context: str, meaning: str, budget: CallBudget,
                 tag: dict, feedback: str = "") -> str:
        """answer() in the English flow: the model writes the sentence, and only the sentence."""
        budget.take()
        if self._client is None:
            self._client = make_client(self.p.generator.api_key)
        prompt = f"Worked examples, on a different database:\n{chain['examples']}\n\nSchema:\n{schema}\n\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"Question: {question}\n\nWhat the user means (confirmed by the user):\n{meaning}"
        if feedback:
            prompt += f"\n\nEarlier attempts had these problems; avoid them:\n{feedback}"
        completion = chat(
            self._client, tag, model=chain.get("model") or self.p.generator.model_name,
            messages=[{"role": "system", "content": f"{ENGLISH_SYSTEM}\n\n{chain['instruction']}"},
                      {"role": "user", "content": prompt}],
            temperature=SQL_TEMPERATURE,
        )
        text = completion.choices[0].message.content or ""
        blocks = re.findall(r"```[a-zA-Z]*\s*(.*?)```", text, re.DOTALL)
        return (blocks[-1] if blocks else text).strip()

    def chain(self, chain: dict, question: str, schema: str, context: str, tables, now,
              budget: CallBudget, tag: dict, retry: bool = False) -> dict:
        """question() then answer(), then Scout's checks, then run; with retry, one more try if rejected."""
        candidate = {"chain": chain["id"], "style": chain["style"], "method": chain.get("method", chain["id"]),
                     "model": chain.get("model"), "meaning": "", "measure": "other", "spec": default_spec(),
                     "round": 1, "sql": None, "sentence": None, "flags": [],
                     "status": "failed", "problems": [], "warnings": []}

        def write(target: dict, feedback: str, call_tag: dict) -> None:
            """Fills in target's SQL: from the model, or in the English flow, from the program."""
            lines = spec_lines(target["spec"])
            meant = target["meaning"] + (f"\nWhat the answer must be:\n{lines}" if lines else "")
            if chain.get("english"):
                target["sentence"] = self.sentence(chain, question, schema, context, meant, budget,
                                                   call_tag, feedback)
                target["sql"] = sentence_to_sql(target["sentence"])
                return
            target["sql"] = self.p.generator.generate(
                question, schema, budget, context=context, feedback=feedback, meaning=meant,
                temperature=SQL_TEMPERATURE, tag=call_tag, instruction=chain["instruction"],
                examples=chain["sql_examples"], model=chain.get("model"),
            )

        sql_tag = {**tag, "stage": "sql", "chain": chain["id"]}
        try:
            meant = self.meaning(chain, question, schema, context, budget, tag)
            text, spec = meant if isinstance(meant, tuple) else (meant, default_spec())
            spec = parse_spec({"measure": spec}) if isinstance(spec, str) else spec
            candidate["meaning"], candidate["spec"] = text, with_question(spec, question)
            candidate["measure"] = candidate["spec"]["measure"]
            write(candidate, "", sql_tag)
        except SentenceRejected as e:
            candidate.update(status="rejected", problems=[f"The program rejected the sentence: {e}"])
        except CallLimitReached as e:
            candidate["problems"] = [str(e)]
            return candidate
        except Exception as e:
            candidate["problems"] = [f"Could not write SQL: {e}"]
            return candidate
        if candidate["sql"]:
            candidate = self.p.check_and_run(candidate, tables, now)
        if retry and candidate["status"] in ("rejected", "failed") and (candidate.get("sql") or candidate.get("sentence")):
            first_problems = candidate["problems"]
            feedback = "\n".join(f"- {p}" for p in first_problems)
            again = {**candidate, "status": "failed", "problems": [], "warnings": [], "retried": True}
            again.pop("error", None)
            try:
                write(again, feedback, {**sql_tag, "round": 2})
                candidate = self.p.check_and_run(again, tables, now)
            except SentenceRejected as e:
                candidate = {**again, "sql": None, "status": "rejected",
                             "problems": [*first_problems, f"The program rejected the new sentence: {e}"]}
            except Exception as e:  # the first try's rejection stands
                candidate["problems"] = [*first_problems, f"The retry failed: {e}"]
        found = self._semantic(candidate) if candidate["status"] == "ok" else []
        if found and retry and not candidate.get("retried"):
            # Version-24: the SQL runs, but doesn't compute or return what the spec says: one more try, told exactly what
            again = {**candidate, "status": "failed", "problems": [], "warnings": [], "retried": True, "flags": []}
            for key in ("error", "rows", "columns", "more"):
                again.pop(key, None)
            try:
                write(again, "\n".join(f"- {message}" for _, message in found), {**sql_tag, "round": 2})
                again = self.p.check_and_run(again, tables, now)
            except Exception:
                again = None
            if again and again["status"] == "ok":
                candidate = again
                found = self._semantic(candidate)
        if found:  # kept, never dropped: it ranks below unflagged answers in the vote
            candidate["flags"] = [code for code, _ in found]
            candidate["warnings"] = [*candidate.get("warnings", []), *(message for _, message in found)]
        if candidate["status"] == "ok":
            rows = candidate["rows"]
            if candidate.get("more"):  # the app keeps the first rows; chains are compared whole
                try:
                    _, rows = self.p.database.run_all(candidate["sql"])
                except Exception:
                    pass
            candidate["answer"] = answer_key(rows)
        return candidate

    def _semantic(self, candidate: dict) -> list:
        return semantic_check.problems(candidate["spec"], candidate["sql"], candidate.get("rows") or [],
                                       getattr(self.p, "database", None), SEMANTIC_CACHE)

    def run(self, question: str, question_id: Optional[str] = None, flow: str = "full") -> dict:
        tables, schema, context, now = self.p._prepare(question)
        adaptive = flow in ADAPTIVE_FLOWS
        all_chains = ENGLISH_CHAINS if flow == ENGLISH_FLOW else CHAINS
        budget = CallBudget(ADAPTIVE_MAX_CALLS if adaptive else FULL_CALLS)
        tag = {"question_id": question_id, "flow": flow}

        def run_chains(group: List[dict]) -> List[dict]:
            with ThreadPoolExecutor(max_workers=len(group)) as pool:
                futures = [pool.submit(self.chain, c, question, schema, context, tables, now, budget, tag, adaptive)
                           for c in group]
                return [f.result() for f in futures]

        if adaptive:
            chains = run_chains(all_chains[:ADAPTIVE_CHAINS])
            if not should_stop_early(chains):  # the same rule as adaptive_vote
                chains += run_chains(all_chains[ADAPTIVE_CHAINS:])
        elif flow == "mixed":
            chains = run_chains(mixed_chains(question))
        else:
            chains = run_chains(CHAINS)

        summary = [{"round": 1, "chain": c["chain"], "style": c["style"], "method": c["method"], "model": c["model"],
                    "meaning": c["meaning"], "sql": c["sql"], "sentence": c.get("sentence"),
                    "status": c["status"], "problems": c["problems"], "answer": c.get("answer"),
                    "retried": bool(c.get("retried")), "measure": c.get("measure"), "spec": c.get("spec"),
                    "flags": c.get("flags", []), "warnings": c.get("warnings", []),
                    "rows": c.get("rows", [])[:3]} for c in chains]
        base = {"llm_calls": budget.used, "attempts": summary}
        unsafe = next((c for c in chains if c["status"] == "unsafe"), None)
        if unsafe:
            return {**base, "sql": unsafe["sql"], "error": "The SQL written could change the data, so it was never run."}
        decision = majority_vote(chains)
        if decision is None:
            last = next((c for c in reversed(chains) if c["sql"]), None)
            return {**base, "sql": last["sql"] if last else None, "error": "No chain produced SQL that passed the checks.",
                    "failure_reason": " ".join(p for c in chains for p in c["problems"])}
        winner = decision["winner"]
        return {**base, "sql": winner["sql"], "columns": winner["columns"], "rows": winner["rows"], "more": winner["more"],
                "meaning": winner["meaning"], "steps": [], "confidence": decision["confidence"],
                "agreeing": decision["agreeing"], "warnings": winner["warnings"]}
