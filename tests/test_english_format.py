"""Version-24: what english2sqlite accepted after the first run. A total can carry a condition or source; division is decimal."""
import sqlite3

import pytest

from backend.english_to_sql.english2sqlite import FormatError, compile_english


def sql(text):
    return compile_english(text)[0].rstrip(";")


@pytest.mark.parametrize("sentence,expected", [
    ('get (count of rows where a is 1) from t', "SELECT (COUNT(*) FILTER (WHERE a = 1)) FROM t"),
    ('get (sum of (a * b) where c is 1 and d is 2) from t', "SELECT (SUM(a * b) FILTER (WHERE c = 1 AND d = 2)) FROM t"),
    ('get (count of distinct a where b is 1) from t', "SELECT (COUNT(DISTINCT a) FILTER (WHERE b = 1)) FROM t"),
    ('get (count of rows from t where a is 1)', "SELECT (SELECT COUNT(*) FROM t WHERE a = 1)"),
    ('get (count of rows where a is 1 from t)', "SELECT (SELECT COUNT(*) FROM t WHERE a = 1)"),
    ('get (sum of x from t join u on t.i = u.i where a is 1 and b is 2)',
     "SELECT (SELECT SUM(x) FROM t JOIN u ON t.i = u.i WHERE a = 1 AND b = 2)"),
    ('get (sum of a from t where b is in (get c from d where e is 1))',
     "SELECT (SELECT SUM(a) FROM t WHERE b IN (SELECT c FROM d WHERE e = 1))"),
])
def test_a_total_can_carry_a_condition_or_source(sentence, expected):
    assert sql(sentence) == expected


def test_only_a_single_total_can():
    for sentence in ("get (a where b is 1) from t", "get (name from t)", "get ((sum of a) + 1 where b is 1) from t"):
        with pytest.raises(FormatError):
            compile_english(sentence)


def test_the_other_columns_still_split_on_and():
    assert sql("get (count of rows where a is 1 and b is 2) and c from t") == \
        "SELECT (COUNT(*) FILTER (WHERE a = 1 AND b = 2)), c FROM t"


def test_division_is_always_decimal_and_the_answer_is_right():
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE TABLE t(a INT); INSERT INTO t VALUES (1), (1), (2), (2), (2), (2);")
    for sentence, value in [
        ("get 1 / 2", 0.5), ("get 1 divided by 2", 0.5), ("get 8 / 2 / 8", 0.5), ("get 2 * 3 / 4", 1.5),
        ("get cast(5 / 2 as integer)", 2),
        ('get (count of rows where a is 1) / (count of rows where a is 2) from t', 0.5),
        ('get (count of rows from t where a is 1) divided by (count of rows from t where a is 2)', 0.5),
    ]:
        assert db.execute(sql(sentence)).fetchone()[0] == value, sentence
