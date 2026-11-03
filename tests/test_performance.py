"""Scout Version-8: size estimate, time limit, row limit, connection pool."""
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.config import MAX_RESULT_ROWS
from backend.text_to_sql.database import QueryTimeout

ENDLESS = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n) SELECT COUNT(*) FROM n"


@pytest.mark.parametrize("sql, expected", [
    ("SELECT * FROM orders WHERE order_id = 5", 1),                           # primary-key lookup
    ("SELECT * FROM orders o JOIN customers c ON c.customer_id = o.customer_id", 50),  # join on a key
    ("SELECT * FROM orders, customers", 2_500),                               # cross join multiplies
    ("SELECT * FROM orders a, orders b, orders c, orders d", 6_250_000),
])
def test_size_estimate(database, row_counts, sql, expected):
    assert database.estimate_reads(sql, row_counts) == expected


def test_with_step_size_is_carried_forward(database, row_counts):
    sql = ("WITH x AS (SELECT store_id, COUNT(*) n FROM orders GROUP BY store_id) "
           "SELECT * FROM x JOIN stores s ON s.store_id = x.store_id")
    assert database.estimate_reads(sql, row_counts) < 1_000


def test_time_limit_stops_an_endless_query(database):
    start = time.monotonic()
    with pytest.raises(QueryTimeout):
        database.query(ENDLESS, timeout=0.5)
    assert time.monotonic() - start < 2


def test_row_limit_and_more_flag(database):
    columns, rows, more = database.query("SELECT a.order_id, b.order_id FROM orders a, orders b")
    assert len(rows) == MAX_RESULT_ROWS and more
    columns, rows, more = database.query("SELECT * FROM stores")
    assert not more


def test_slow_query_does_not_block_fast_ones(database):
    def fast():
        start = time.monotonic()
        database.query("SELECT COUNT(*) FROM orders")
        return time.monotonic() - start

    with ThreadPoolExecutor(3) as pool:
        slow = pool.submit(lambda: pytest.raises(QueryTimeout, database.query, ENDLESS, timeout=1.5))
        time.sleep(0.2)  # the slow query has started
        fast_times = [pool.submit(fast).result() for _ in range(2)]
        slow.result()
    assert max(fast_times) < 0.5
