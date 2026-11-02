# Text-to-SQL Practice Database

A small retail company database for testing a text-to-SQL system before running it on BIRD.

## Files

| File | What it is |
|---|---|
| `retail_store.sqlite` | The database: 10 tables, 50 rows each, all in one file so joins work |
| `schema.sql` | `CREATE TABLE` statements, with comments on units and formats |
| `schema.json` | The same schema in JSON (columns, types, foreign keys), easy to load in Python |
| `questions.md` | 20 questions (6 easy, 8 medium, 6 hard) plus an answer key with correct SQL and results |
| `questions.json` | The same questions in JSON, ready for an evaluation script |

## Tables

`stores`, `employees`, `customers`, `suppliers`, `products`, `orders`, `order_items`, `payments`, `shipments`, `reviews`

## Traps built into the data

These match the failure types from the design discussion:

- **Messy values:** New York City appears in `customers.city` as `New York`, `NY`, `NYC` and `new york`.
- **Hidden date format:** `orders.order_date` is TEXT in `MM/DD/YYYY` format. Every other date is `YYYY-MM-DD`.
- **Units:** all prices and amounts are stored in cents (`unit_price_cents`, `amount_cents`).
- **Cancelled orders:** they have no `order_items` and their payments are `refunded`, so revenue questions must exclude them.

A system that only sees the schema will miss the first three. Sending sample values in the prompt fixes them.

## Quick start

```python
import json, sqlite3

conn = sqlite3.connect("retail_store.sqlite")
questions = json.load(open("questions.json"))

q = questions[0]
print(q["question"])
print(conn.execute(q["gold_sql"]).fetchall())
```

To score your system: run its SQL and `gold_sql` on the database, then compare the two result sets.
