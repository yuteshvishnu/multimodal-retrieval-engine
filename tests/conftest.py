"""
Shared setup for the tests. They need no API key and make no LLM calls: they check the
deterministic rules (safety, performance limits, fingerprints, caching, document reading)
against the practice database in data/text_to_sql/retail_store.

    python -m pytest
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # the app reads data/ relative to the project root


@pytest.fixture(scope="session")
def database():
    from backend.text_to_sql.database import SQLiteDatabase

    return SQLiteDatabase()


@pytest.fixture(scope="session")
def row_counts(database):
    return {t["name"].lower(): t["row_count"] for t in database.tables}
