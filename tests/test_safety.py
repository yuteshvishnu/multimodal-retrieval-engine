"""Scout Version-7: anything that could change a database is caught, normal reads are not."""
import pytest

from backend.text_to_sql.safety import write_problems, write_request

WRITE_SQL = [
    "SELECT 1; DROP TABLE orders",
    "WITH x AS (SELECT 1) DELETE FROM orders",
    "PRAGMA writable_schema=1",
    "ATTACH 'x.db' AS y",
    "SELECT * INTO new_t FROM orders",
    "REPLACE INTO orders VALUES (1)",
    "INSERT OR REPLACE INTO t VALUES (1)",
    "VACUUM",
    "BEGIN",
    "UPDATE orders SET status='x'",
    "SELECT load_extension('x')",
    "CREATE TEMP TABLE t AS SELECT 1",
    "DETACH y",
    "REINDEX",
    "ANALYZE",
    "DELETE FROM orders WHERE 1=1 -- harmless",
    "select * from orders;\n update orders set status = 'x'",
]

READ_SQL = [
    "SELECT * FROM orders WHERE status = 'deleted'",
    "SELECT replace(name, 'a', 'b') FROM customers",
    "WITH c AS (SELECT store_id, COUNT(*) n FROM orders GROUP BY store_id) SELECT * FROM c ORDER BY n DESC LIMIT 5",
    "SELECT a FROM t UNION SELECT b FROM u",
    "SELECT update_date, created_at FROM t",
    "SELECT 'drop table x' AS note;",
    "SELECT RANK() OVER (PARTITION BY s ORDER BY n DESC) FROM t",
]

WRITE_QUESTIONS = [
    "Delete all cancelled orders",
    "Please remove customer 5",
    "Can you update the price of product 3 to $10?",
    "I want to add a new store in Austin",
    "Let's rename the orders table",
    "Set every order status to delivered",
    "Increase all prices by 10%",
    "DELETE FROM orders",
    "drop table customers",
    "Could you please clear the reviews?",
    "update orders set status='x'",
    "Insert a review for product 2",
]

READ_QUESTIONS = [
    "How many orders were placed in March 2026?",
    "Which customers were added in 2026?",
    "Add up the revenue for each store",
    "Show the orders that were updated last week",
    "Which products were removed from stock?",
    "List deleted orders",
    "Select the top 5 customers by spend",
    "With the cancelled orders included, how many orders did each store get?",
    "What is the average price change?",
    "Count orders by status",
]


@pytest.mark.parametrize("sql", WRITE_SQL)
def test_write_sql_is_blocked(sql):
    assert write_problems(sql)


@pytest.mark.parametrize("sql", READ_SQL)
def test_read_sql_passes(sql):
    assert write_problems(sql) == []


@pytest.mark.parametrize("question", WRITE_QUESTIONS)
def test_write_request_is_blocked(question):
    assert write_request(question)


@pytest.mark.parametrize("question", READ_QUESTIONS)
def test_read_question_passes(question):
    assert write_request(question) is None


def test_database_refuses_writes_even_if_checks_were_skipped(database):
    with pytest.raises(Exception):
        database.query("DELETE FROM orders")
    with database._connection() as conn, pytest.raises(Exception):
        conn.execute("DELETE FROM orders")  # attached read-only + PRAGMA query_only
