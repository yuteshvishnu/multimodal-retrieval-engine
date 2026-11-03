"""Scout Version-8: comparing answers by fingerprint, and the LRU result cache."""
from backend.text_to_sql.checks import result_signature
from backend.text_to_sql.results import ResultCache


def test_fingerprint_ignores_row_order_and_float_noise():
    assert result_signature([[1, "a"], [2, 0.1 + 0.2]]) == result_signature([[2, 0.3], [1, "a"]])


def test_fingerprint_sees_any_difference():
    assert result_signature([[1], [2]]) != result_signature([[1], [3]])
    assert result_signature([[1], [1]]) != result_signature([[1]])  # duplicates count


def test_cache_drops_least_recently_used():
    cache = ResultCache(capacity=2)
    a = cache.put("SELECT 1", ["x"], [[1]], False)
    b = cache.put("SELECT 2", ["x"], [[2]], False)
    cache.get(a)                                   # a is now the most recently used
    cache.put("SELECT 3", ["x"], [[3]], False)     # so b is dropped
    assert cache.get(a) is not None and cache.get(b) is None
    assert cache.sql_for(b) == "SELECT 2"          # its SQL is remembered, to rebuild it
