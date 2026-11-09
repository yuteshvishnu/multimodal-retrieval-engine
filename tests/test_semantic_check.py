"""Version-24: the structured meaning and the code checks against it. A small made-up database, no LLM."""
import sqlite3

import pytest

from backend.text_to_sql import flows
from backend.text_to_sql.semantic_check import default_spec, parse_spec, problems, spec_lines, with_question


class DB:
    def __init__(self):
        self.c = sqlite3.connect(":memory:", check_same_thread=False)
        self.c.executescript("""
            CREATE TABLE usage(customer TEXT, month TEXT, amount REAL);
            INSERT INTO usage VALUES ('a','201301',5),('b','201301',7),('a','201302',9),('b','201302',1);
            CREATE TABLE months(month TEXT, total REAL);
            INSERT INTO months VALUES ('201301',12),('201302',10);
        """)

    def run_all(self, sql, timeout=30):
        cur = self.c.execute(sql)
        return [d[0] for d in cur.description], [list(r) for r in cur.fetchall()]


def codes(spec, sql, db):
    _, rows = db.run_all(sql)
    return [code for code, _ in problems(spec, sql, rows, db, {})]


@pytest.fixture
def db():
    return DB()


def test_the_spec_is_cleaned_and_filled_from_the_question():
    assert parse_spec({"extreme": "Highest", "grain": "Month", "date_part": "week", "values_per_row": "3"}) == {
        "measure": "other", "grain": "month", "extreme": "highest", "date_part": "none", "values_per_row": 3}
    assert parse_spec({"values_per_row": 1})["values_per_row"] == 0 and default_spec()["grain"] == "none"
    assert with_question(default_spec(), "What was the peak month for sales?")["date_part"] == "month"
    assert with_question(default_spec(), "How many sales last month? (Hint: which month = ...)")["date_part"] == "none"
    assert with_question(default_spec(), "How many customers have at least one month with sales over 10?")["date_part"] == "none"
    assert "Return the month only" in spec_lines({**default_spec(), "date_part": "month"})


def test_a_whole_date_where_a_month_is_asked_is_flagged(db):
    spec = {**default_spec(), "date_part": "month"}
    assert codes(spec, "SELECT month FROM months", db) == ["OUTPUT_DATE_PART"]
    assert codes(spec, "SELECT substr(month, 5, 2) FROM months", db) == []
    assert codes(default_spec(), "SELECT month FROM months", db) == []


def test_n_rows_where_one_row_of_n_values_is_asked_is_flagged(db):
    spec = {**default_spec(), "values_per_row": 2}
    assert codes(spec, "SELECT SUM(amount) FROM usage GROUP BY customer", db) == ["OUTPUT_SHAPE"]
    labelled = {**default_spec(), "values_per_row": 3}  # as in #1482: a label and a value per row, 3 rows
    _, rows = db.run_all("SELECT 'a', 1 UNION ALL SELECT 'b', 2 UNION ALL SELECT 'c', 3")
    assert [c for c, _ in problems(labelled, "SELECT 1", rows, db, {})] == ["OUTPUT_SHAPE"]
    assert codes(spec, "SELECT SUM(amount) FILTER (WHERE customer='a'), SUM(amount) FILTER (WHERE customer='b') FROM usage", db) == []


def test_the_highest_of_single_rows_is_flagged_only_when_rows_repeat_per_unit(db):
    spec = {**default_spec(), "extreme": "highest", "grain": "month"}
    assert codes(spec, "SELECT MAX(amount) FROM usage WHERE month LIKE '2013%'", db) == ["AGGREGATION_GRAIN"]
    assert codes(spec, "SELECT amount FROM usage WHERE month LIKE '2013%' ORDER BY amount DESC LIMIT 1", db) == ["AGGREGATION_GRAIN"]
    assert codes(spec, "SELECT SUM(amount) FROM usage GROUP BY month ORDER BY 1 DESC LIMIT 1", db) == []
    assert codes(spec, "SELECT MAX(total) FROM months WHERE month LIKE '2013%'", db) == []  # one row per month already
    per_customer = {**default_spec(), "extreme": "highest", "grain": "customer"}
    assert codes(per_customer, "SELECT MAX(amount) FROM usage", db) == ["AGGREGATION_GRAIN"]


def test_flagged_answers_rank_last_and_stop_nothing_early():
    flagged = {"chain": "a", "status": "ok", "answer": "x", "rows": [[1]], "flags": ["OUTPUT_DATE_PART"]}
    other = {"chain": "b", "status": "ok", "answer": "y", "rows": [[2]], "flags": []}
    assert flows.majority_vote([flagged, {**flagged, "chain": "c"}, other])["winner"]["chain"] == "b"
    assert flows.majority_vote([flagged])["winner"]["chain"] == "a"  # never dropped
    assert not flows.should_stop_early([flagged, {**flagged, "chain": "c"}])
    assert flows.should_stop_early([{**flagged, "flags": []}, {**flagged, "chain": "c", "flags": []}])


def test_the_meaning_examples_carry_the_whole_spec():
    import json
    for chain in flows.CHAINS:
        for line in chain["meaning_examples"].splitlines():
            if line.startswith("{"):
                assert set(json.loads(line)) == {"meaning", "measure", "grain", "extreme", "date_part", "values_per_row"}


def test_an_empty_lookup_on_a_value_the_data_holds_elsewhere_is_flagged(db):
    db.c.executescript("CREATE TABLE pay(customer TEXT, qty INT, price REAL); INSERT INTO pay VALUES ('a', 2, 548.4);")
    assert codes(default_spec(), "SELECT customer FROM pay WHERE qty * price = 548.4", db) == ["VALUE_GROUNDING"]
    assert codes(default_spec(), "SELECT customer FROM pay WHERE price = 548.4", db) == []  # found: nothing to say
    assert codes(default_spec(), "SELECT customer FROM pay WHERE price = 999.5", db) == []  # nowhere in the data
    message = problems(default_spec(), "SELECT customer FROM pay WHERE qty * price = 548.4", [], db, {})[0][1]
    assert "pay.price (1 rows)" in message


def test_dividing_two_whole_numbers_is_flagged(db):
    assert codes(default_spec(), "SELECT COUNT(*) / COUNT(DISTINCT customer) FROM usage", db) == ["NUMERIC_TYPE"]
    assert codes(default_spec(), "SELECT SUM(CASE WHEN amount > 5 THEN 1 ELSE 0 END) / COUNT(*) FROM usage", db) == ["NUMERIC_TYPE"]
    for fine in ("SELECT COUNT(*) * 1.0 / COUNT(DISTINCT customer) FROM usage", "SELECT SUM(amount) / COUNT(*) FROM usage",
                 "SELECT CAST(COUNT(*) AS REAL) / 2 FROM usage"):
        assert codes(default_spec(), fine, db) == []
