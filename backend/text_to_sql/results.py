"""
Scout Version-8: keeps recent query results on the server, so the UI can load one page
(10 rows) at a time without running the query again.

LRU (least recently used): the 10 most recently used results are kept; using a result
moves it to the front, and adding an 11th drops the one unused the longest. The SQL of
older results is remembered (just the text), so a dropped result can be run again once.
"""
import threading
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional

SQL_REMEMBERED = 1000  # result id -> SQL, so an evicted result can be rebuilt


class ResultCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._results: "OrderedDict[str, dict]" = OrderedDict()
        self._sql: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, sql: str, columns: List[str], rows: List[list], more: bool, result_id: Optional[str] = None) -> str:
        result_id = result_id or uuid.uuid4().hex[:12]
        with self._lock:
            self._results[result_id] = {"sql": sql, "columns": columns, "rows": rows, "more": more}
            self._results.move_to_end(result_id)
            while len(self._results) > self.capacity:
                self._results.popitem(last=False)  # drop the least recently used
            self._sql[result_id] = sql
            self._sql.move_to_end(result_id)
            while len(self._sql) > SQL_REMEMBERED:
                self._sql.popitem(last=False)
        return result_id

    def get(self, result_id: str) -> Optional[Dict]:
        with self._lock:
            result = self._results.get(result_id)
            if result is not None:
                self._results.move_to_end(result_id)  # now the most recently used
            return result

    def sql_for(self, result_id: str) -> Optional[str]:
        with self._lock:
            return self._sql.get(result_id)
