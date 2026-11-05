"""
Version-20: the schema catalog (offline onboarding), the value index, and how they reach the LLM
and the checks. A small database made here, no LLM.
"""
import datetime
import sqlite3
import types

import pytest

from backend.text_to_sql import column_notes, schema_catalog
from backend.text_to_sql.checks import check_sql
from backend.text_to_sql.database import SQLiteDatabase


@pytest.fixture
def shop(tmp_path, monkeypatch):
    """A database with one column of every kind, a declared link and an undeclared one."""
    monkeypatch.setattr(schema_catalog, "CATALOG_DIR", tmp_path / "catalogs")
    folder = tmp_path / "data" / "shop"
    folder.mkdir(parents=True)
    con = sqlite3.connect(folder / "shop.sqlite")
    con.executescript("""
        CREATE TABLE stations (station_id INTEGER PRIMARY KEY, country TEXT, segment TEXT, opened TEXT);
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, segment TEXT, email TEXT, city TEXT,
                                notes TEXT, nickname TEXT);
        CREATE TABLE visits (visit_id INTEGER PRIMARY KEY,
                             customer_id INTEGER REFERENCES customers(customer_id),
                             station_id INTEGER, month TEXT, liters REAL, rating INTEGER);
    """)
    segments = ["Discount", "Premium", "Value for money"]
    cities = ["Springfield", "Shelbyville", "Capital City"] + [f"Town {i}" for i in range(30)]
    for i in range(1, 61):
        con.execute("INSERT INTO stations VALUES (?, ?, ?, ?)",
                    (i, "CZE" if i % 3 else "SVK", segments[i % 3], f"2020-01-{1 + i % 28:02d}"))
        con.execute("INSERT INTO customers VALUES (?, ?, ?, ?, ?, NULL)",
                    (i, ["KAM", "SME"][i % 2], f"person{i}@example.com", cities[i % len(cities)], "x" * 120))
    for i in range(1, 201):
        con.execute("INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?)",
                    (i, 1 + i % 60, 1 + i % 60, f"2013{1 + i % 12:02d}", i * 1.5, 1 + i % 5))
    con.commit()
    con.close()
    db = SQLiteDatabase(str(tmp_path / "data"))
    yield db
    db.close()


def columns(db, table):
    entry = next(t for t in db.tables if t["name"] == table)["catalog"]
    return {c["name"]: c for c in entry["columns"]}


def test_each_column_gets_the_profile_its_kind_needs(shop):
    stations, customers, visits = columns(shop, "stations"), columns(shop, "customers"), columns(shop, "visits")
    assert stations["segment"]["kind"] == "category"
    assert sorted(v for v, _ in stations["segment"]["values"]) == ["Discount", "Premium", "Value for money"]
    assert customers["city"]["kind"] == "text" and customers["city"]["top"][0][1] > 1  # repeats: most common
    assert customers["email"]["kind"] == "text" and len(customers["email"]["examples"]) == 3  # all unique: examples
    assert customers["notes"]["kind"] == "long_text" and "values" not in customers["notes"]
    assert customers["nickname"]["kind"] == "empty"
    assert visits["liters"]["kind"] == "number" and (visits["liters"]["min"], visits["liters"]["max"]) == (1.5, 300)
    assert visits["rating"]["kind"] == "number" and len(visits["rating"]["values"]) == 5  # few numbers: listed
    assert stations["opened"]["date_format"] == "YYYY-MM-DD"
    assert visits["month"]["time_format"] == "YYYYMM" and visits["month"]["min"] == "201301"
    assert visits["station_id"]["kind"] == "key"


def test_links_are_declared_or_inferred_from_the_data(shop):
    links = next(t for t in shop.tables if t["name"] == "visits")["catalog"]["links"]
    declared = [l for l in links if not l["inferred"]]
    inferred = [l for l in links if l["inferred"]]
    assert [(l["column"], l["table"]) for l in declared] == [("customer_id", "customers")]
    assert [(l["column"], l["table"], l["to"]) for l in inferred] == [("station_id", "stations", "station_id")]
    assert next(t for t in shop.tables if t["name"] == "visits")["links"] == {"customers", "stations"}


def test_the_llm_sees_meaning_and_contents_of_every_column(shop):
    schema = next(t for t in shop.tables if t["name"] == "stations")["schema"]
    assert "segment TEXT,  -- values: 'Discount'" in schema or "values: 'Discount', 'Premium', 'Value for money'" in schema
    assert "format YYYY-MM-DD, from 2020-01-01 to 2020-01-28" in schema
    visits = next(t for t in shop.tables if t["name"] == "visits")["schema"]
    assert "links to stations.station_id (not declared" in visits
    assert "REFERENCES customers(customer_id)" in visits and "from 1.5 to 300" in visits
    # SHARE_ROWS_WITH_LLM off: no values or examples, but formats and ranges stay
    private = schema_catalog.render_schema("stations", next(t for t in shop.tables if t["name"] == "stations")["catalog"],
                                           False, lambda n: n)
    assert "Discount" not in private and "3 different values" in private and "from 2020-01-01" in private


def test_onboarding_happens_once_and_again_only_when_the_database_changes(shop, tmp_path, capsys):
    path = shop.files[0]
    shop.tables  # first use: onboarded now
    assert schema_catalog.load(path, "shop") is not None  # and saved
    capsys.readouterr()
    SQLiteDatabase(str(tmp_path / "data")).tables
    assert "[Catalog]" not in capsys.readouterr().out  # read from disk, not built again
    con = sqlite3.connect(path)
    con.execute("INSERT INTO stations VALUES (999, 'AUT', 'Discount', '2021-05-05')")
    con.commit()
    con.close()
    assert schema_catalog.load(path, "shop") is None  # the file changed: out of date
    again = SQLiteDatabase(str(tmp_path / "data"))
    assert "'AUT'" in next(t for t in again.tables if t["name"] == "stations")["schema"]


def test_question_words_find_the_stored_values(shop):
    found = shop.values.match("How many discount stations are in each country?", [("shop", "stations")])
    assert (found[0]["table"], found[0]["column"], found[0]["value"]) == ("stations", "segment", "Discount")
    assert not shop.values.match("How many discount stations?", [("shop", "customers")])  # other tables: not searched
    assert shop.values.match("value for money stations")[0]["value"] == "Value for money"


def test_a_value_the_data_lacks_is_caught_in_any_text_column(shop):
    today = datetime.date.today()
    errors, _ = check_sql("SELECT * FROM customers WHERE city = 'Springfeld'", shop.tables, today, shop.values)
    assert errors == ["city has no value 'Springfeld'; similar values: 'Springfield'."]  # 33 cities: too many to list
    assert check_sql("SELECT * FROM customers WHERE city = 'Springfield'", shop.tables, today, shop.values)[0] == []
    # segment is in two tables: only the one the SQL reads counts
    errors, _ = check_sql("SELECT * FROM stations WHERE segment = 'SME'", shop.tables, today, shop.values)
    assert errors and "'Discount'" in errors[0] and "'KAM'" not in errors[0]


def test_ai_notes_fill_only_undescribed_columns_and_are_marked(shop, tmp_path):
    def create(**kwargs):
        content = '{"columns": {"country": "Country code of the station", "segment": "unclear", "ghost": "x"}}'
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
                                     usage=None)

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    calls = column_notes.ColumnNoteWriter(client).write(shop)
    assert calls == 3  # one per table with undescribed columns
    notes = column_notes.load(shop.files[0], "shop")
    assert notes["stations"] == {"country": "Country code of the station"}  # "unclear" and unknown names dropped
    fresh = SQLiteDatabase(str(tmp_path / "data"))
    schema = next(t for t in fresh.tables if t["name"] == "stations")["schema"]
    assert "Country code of the station (AI note)" in schema


def test_llm_summaries_stay_valid_when_the_profile_changes(shop):
    from backend.text_to_sql.table_summaries import fingerprint

    table = next(t for t in shop.tables if t["name"] == "stations")
    assert fingerprint(table) == fingerprint({**table, "schema": table["schema"] + " -- new profile"})


def test_grouping_by_a_shared_name_is_caught_but_categories_are_fine(tmp_path, monkeypatch):
    """Practice Q9: two customers called Ananya Kim were added together by GROUP BY first_name, last_name."""
    from backend.text_to_sql.checks import merged_records

    monkeypatch.setattr(schema_catalog, "CATALOG_DIR", tmp_path / "catalogs")
    folder = tmp_path / "data" / "people"
    folder.mkdir(parents=True)
    con = sqlite3.connect(folder / "people.sqlite")
    con.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, city TEXT);
        CREATE TABLE payments (payment_id INTEGER PRIMARY KEY,
                               customer_id INTEGER REFERENCES customers(customer_id), amount REAL);
    """)
    for i in range(1, 41):
        first, last = ("Ananya", "Kim") if i in (28, 39) else (f"First{i}", f"Last{i}")
        con.execute("INSERT INTO customers VALUES (?, ?, ?, ?)", (i, first, last, f"City{i % 4}"))
        con.execute("INSERT INTO payments VALUES (?, ?, ?)", (i, i, 10.0 * i))
    con.commit()
    con.close()
    db = SQLiteDatabase(str(tmp_path / "data"))
    join = "FROM payments p JOIN customers c ON p.customer_id = c.customer_id"

    problems = merged_records(f"SELECT c.first_name, c.last_name, SUM(p.amount) {join} GROUP BY c.first_name, c.last_name",
                              db.tables, db.run)
    assert len(problems) == 1 and "'Ananya Kim' is more than one record" in problems[0]
    assert "Group by customers.customer_id too" in problems[0]
    for fine in (
        f"SELECT c.first_name, c.last_name, SUM(p.amount) {join} GROUP BY c.customer_id, c.first_name, c.last_name",
        f"SELECT c.first_name, c.last_name, SUM(p.amount) {join} GROUP BY p.customer_id, c.first_name, c.last_name",
        f"SELECT c.city, SUM(p.amount) {join} GROUP BY c.city",  # a category: adding up is the point
        "WITH t AS (SELECT customer_id, amount FROM payments) SELECT customer_id, SUM(amount) FROM t GROUP BY customer_id",
    ):
        assert merged_records(fine, db.tables, db.run) == [], fine
    db.close()
