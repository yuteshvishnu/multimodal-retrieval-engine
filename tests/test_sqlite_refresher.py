"""
Version-23: the SQLite refresher. Every example in it must really run, pass Scout's read-only check, and
contain nothing from BIRD; it must reach the calls that write SQL and no others. No LLM calls.
"""
import re
import sqlite3
import types

import pytest

from backend.text_to_sql import flows
from backend.text_to_sql.flows import FlowRunner
from backend.text_to_sql.safety import write_problems
from backend.text_to_sql.sql_generator import FINAL_RULE, REFRESHER, CallBudget, SQLGenerator

SHOP = """
CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, city TEXT, phone TEXT, signup_date TEXT);
CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT, total REAL);
CREATE TABLE order_items(order_id INTEGER, product_id INTEGER, qty INTEGER, unit_price REAL);
INSERT INTO customers VALUES (1, 'Ana Kim', 'Austin', '555-0101', '2020-05-01'), (2, 'Jon Lee', 'Boston', NULL, '2021-06-02');
INSERT INTO products VALUES (1, 'Kite', 'Toys', 10.5), (2, 'Atlas', 'Books', 20), (3, 'Yo-yo', 'Toys', 0);
INSERT INTO orders VALUES (1, 1, '2023-03-15', 'paid', 120), (2, 1, '2024-03-16', 'open', 40), (3, 2, '2024-04-01', 'paid', 160);
INSERT INTO order_items VALUES (1, 1, 2, 10.5), (1, 2, 1, 20), (3, 3, 4, 1);
"""
EXAMPLES = re.findall(r"```sql\n(.*?)```", REFRESHER, re.DOTALL)


def test_the_refresher_has_every_kind_of_command():
    assert len(EXAMPLES) >= 50
    for word in ("LEFT JOIN", "HAVING", "RANK()", "UNION", "EXCEPT", "WITH RECURSIVE", "STRFTIME", "JULIANDAY", "CAST",
                 "COALESCE", "NULLIF", "IIF", "OVER", "LAG", "SUBSTR", "INSTR", "EXISTS", "GROUP_CONCAT"):
        assert word in REFRESHER, word


@pytest.mark.parametrize("sql", EXAMPLES)
def test_every_example_runs_and_is_read_only(sql):
    db = sqlite3.connect(":memory:")
    db.executescript(SHOP)
    db.execute(sql).fetchall()  # raises if SQLite can't run it
    assert not write_problems(sql)


def test_the_refresher_shares_nothing_with_birds_databases():
    for word in ("BIRD", "superhero", "thrombosis", "debit_card", "yearmonth", "gasstations", "student_club", "formula_1",
                 "football", "Consumption"):
        assert word.lower() not in REFRESHER.lower(), word


def generator(seen):
    def create(**kwargs):
        seen.append(kwargs)
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
        message = types.SimpleNamespace(content="SELECT 1")
        return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(message=message)])

    made = SQLGenerator()
    made.api_key = "k"
    made.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return made


def test_sql_calls_get_it_and_a_chains_instruction_still_applies():
    seen = []
    made = generator(seen)
    made.refresher = True
    made.generate("Q", "schema", CallBudget(2), instruction="Write notes, then a sql block.", examples="ex")
    system = seen[0]["messages"][0]["content"]
    assert REFRESHER in system and "Write notes, then a sql block." in system and FINAL_RULE not in system
    assert system.index("Write notes") < system.index(REFRESHER)  # Scout's own rules come first


def test_it_can_be_turned_off():
    seen = []
    made = generator(seen)
    made.refresher = False
    made.generate("Q", "schema", CallBudget(1))
    assert REFRESHER not in seen[0]["messages"][0]["content"]


def test_the_meaning_call_never_gets_it():
    seen = []
    made = generator(seen)
    made.refresher = True
    runner = FlowRunner(types.SimpleNamespace(generator=made))
    runner._client = made.client
    chain = flows.mixed_chains("How many?")[0]
    runner.meaning(chain, "How many?", "schema", "", CallBudget(1), {"stage": "meaning"})
    assert all(REFRESHER not in m["content"] for m in seen[0]["messages"])
