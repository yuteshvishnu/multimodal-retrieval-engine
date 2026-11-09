# The English format

Each sentence compiles to exactly one SQLite command. Nothing is guessed: a sentence that doesn't fit, or could mean two things, is an error that points at the problem.

```bash
python3 english2sqlite.py 'get name from users where age is at least 18'
```

Add `--db file.db` to also run the commands and print the results. Writes really change the file. Use `-f file.txt` to read a script, or give no arguments for an interactive prompt.

## Basic rules

- Case doesn't matter for format words. The word `the` is ignored before values and names.
- `"double"` or `'single'` quotes are always text. To include the quote character, double it.
- `` `backticks` `` are always a name. Use them for names with spaces or that match a format word: `` `first name` ``, `` `order` ``.
- In a script, put one sentence per line, or separate sentences with `;`. An indented line continues the previous sentence. `#` and `--` start comments.
- Clauses of a query can come in any order, and the SQL always comes out in the correct order.

## Queries

Start with any of: `get`, `select`, `show`, `find`, `list`, `fetch`, `retrieve`, `display`, `return`, `give me`, `what is`, `what are`.

| Part | Write | SQL |
|---|---|---|
| Columns | `name, age` or `name and age` | `name, age` |
| All columns | `everything`, `all columns`, `*` | `*` |
| No duplicates | `get unique city` | `SELECT DISTINCT city` |
| Alias | `x as y`, `x called y`, `x named y` | `x AS y` |
| Source | `from users as u`, `from (get ...) as t` | `FROM ...` |
| Join | `join`, `inner join`, `left join`, `right join`, `full join`, `cross join`, `natural join`, then `on ...` or `using (a)` | `... JOIN ...` |
| Filter | `where ...`, `whose ...` | `WHERE ...` |
| Group | `group by x`, `grouped by x`, `for each x`, `per x` | `GROUP BY x` |
| Group filter | `having ...` | `HAVING ...` |
| Sort | `sorted by`, `sort by`, `ordered by`, `order by`; separate with `,`, `and`, `then` | `ORDER BY ...` |
| Direction | `ascending`, `lowest first`, `oldest first`, `a to z` / `descending`, `highest first`, `newest first`, `z to a` | `ASC` / `DESC` |
| Nulls | `nulls first`, `nulls last` | `NULLS FIRST/LAST` |
| Limit | `limit 5`, `limited to 5`, `get top 5 name` | `LIMIT 5` |
| Skip | `skip 5`, `offset 5` | `OFFSET 5` |
| Combine | `union`, `union all`, `intersect`, `except` | same |
| Named query | `with a as (get ...) get ... from a`, `with recursive ...` | `WITH ...` |
| Count shortcut | `how many users where ...`, `how many users are there`, `how many unique city from users` | `SELECT COUNT(...)` |

## Conditions

Combine them with `and`, `or`, `not` and parentheses. `and` binds tighter than `or`, as in SQL.

| Write | SQL |
|---|---|
| `x is 5`, `x equals 5`, `x = 5` | `x = 5` |
| `x is not 5`, `x does not equal 5`, `x != 5` | `x <> 5` |
| `x is greater than 5`, `more than`, `above`, `over`, `after`, `exceeds` | `x > 5` |
| `x is less than 5`, `fewer than`, `below`, `under`, `before` | `x < 5` |
| `x is at least 5`, `no less than`, `greater than or equal to` | `x >= 5` |
| `x is at most 5`, `no more than`, `less than or equal to` | `x <= 5` |
| `x is between 1 and 5`, `x is not between 1 and 5` | `x [NOT] BETWEEN 1 AND 5` |
| `x is one of (1, 2)`, `x is in (get ...)`, `x is not in (1, 2)` | `x [NOT] IN (...)` |
| `x is null`, `x is missing`, `x has no value` | `x IS NULL` |
| `x is not null`, `x is not missing`, `x has a value` | `x IS NOT NULL` |
| `x is empty` | `x = ''` |
| `x is true`, `x is false` | `x IS TRUE`, `x IS FALSE` |
| `x contains "a"` | `x LIKE '%a%'` |
| `x starts with "a"`, `x begins with "a"` | `x LIKE 'a%'` |
| `x ends with "a"` | `x LIKE '%a'` |
| `x does not contain "a"` (also `start with`, `end with`, `match`) | `x NOT LIKE ...` |
| `x matches "a%"`, `x like "a%"` (you supply the wildcards) | `x LIKE 'a%'` |
| `x glob "a*"`, `x regexp "..."` | `x GLOB ...`, `x REGEXP ...` |
| `x is distinct from y` | `x IS DISTINCT FROM y` |
| `exists (get ...)`, `not exists (get ...)` | `[NOT] EXISTS (...)` |
| `x is "a" ignoring case`, `x collate nocase` | `... COLLATE NOCASE` |

In `contains`, `starts with` and `ends with`, a `%` or `_` in quoted text is treated as a literal character. Like all SQLite `LIKE` matching, these ignore case for ASCII letters by default.

## Values

| Write | SQL |
|---|---|
| `count of rows`, `number of rows`, `count of everything` | `COUNT(*)` |
| `count of x`, `count of unique x` | `COUNT(x)`, `COUNT(DISTINCT x)` |
| `sum of x`, `total of x` | `SUM(x)` |
| `average of x`, `avg of x`, `mean of x` | `AVG(x)` |
| `maximum of x`, `max of x`, `highest x`, `largest x` | `MAX(x)` |
| `minimum of x`, `min of x`, `lowest x`, `smallest x` | `MIN(x)` |
| `grouped list of x separated by ", "` | `GROUP_CONCAT(x, ', ')` |
| `uppercase of x`, `lowercase of x`, `length of x`, `trim of x`, `absolute value of x` | `UPPER`, `LOWER`, `LENGTH`, `TRIM`, `ABS` |
| `substring of x from 1 for 3` | `SUBSTR(x, 1, 3)` |
| `year of d` (also `month`, `day`, `hour`, `minute`, `second`, `weekday`) | `CAST(strftime('%Y', d) AS INTEGER)` |
| `date of d`, `time of d` | `DATE(d)`, `TIME(d)` |
| `today`, `current date` / `now`, `current timestamp` / `current time` | `CURRENT_DATE` / `CURRENT_TIMESTAMP` / `CURRENT_TIME` |
| `7 days ago`, `2 weeks ago`, `1 year from now` | `date('now', '-7 days')`, ... |
| `3 hours ago` (also `minutes`, `seconds`) | `datetime('now', '-3 hours')` |
| `a plus b`, `minus`, `times`, `modulo`, or `+ - * %` | arithmetic |
| `a divided by b`, `a / b` | `a * 1.0 / b`: always a decimal, so `1 / 2` is 0.5. For a whole number: `cast(a / b as integer)` |
| `a concatenated with b`, `a followed by b`, `a \|\| b` | `a \|\| b` |
| `a / b rounded to 2 places` | `ROUND(a / b, 2)` |
| `x converted to integer`, `cast(x as integer)` | `CAST(x AS INTEGER)` |
| `if c then a else b`, `case when c then a else b end` | `CASE WHEN ...` |
| `(get max of x from t)` | a subquery |
| `any_function(a, b)`, `count(distinct x)` | passed through as a function call |
| `raw "x ->> '$.a'"` | inserted into the SQL unchanged |

Window functions and filters:

- `rank over (per city sorted by age descending)` gives `RANK() OVER (PARTITION BY city ORDER BY age DESC)`. `row number`, `dense rank` and `percent rank` also work.
- `sum of x over (sorted by d frame "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW")` adds a window frame.
- `count of rows filter (where x is over 5)` gives `COUNT(*) FILTER (WHERE x > 5)`.

A total in parentheses can carry its own condition or source:

- `(count of rows where x is over 5)` gives `(COUNT(*) FILTER (WHERE x > 5))`, over the query's own rows.
- `(sum of x from t where y is 1)` gives `(SELECT SUM(x) FROM t WHERE y = 1)`, a query of its own, with any clauses in any order.

### Two rules that prevent wrong answers

**1. A phrase and arithmetic need parentheses.** `sum of price * qty` could mean two different things, so it's an error. Write what you mean:

- `sum of (price * qty)` gives `SUM(price * qty)`
- `(sum of price) * qty` gives `(SUM(price)) * qty`

**2. In a column list, `and` separates items.** `get a and b` returns two columns. For a boolean value, use parentheses: `get (a and b)`. The same applies to `set`, `group by` and `sorted by`.

## Changing data

```text
add to users: name = "Ana", age = 34
add a row to users with name is "Ana" and age is 34
insert into users (name, age) values ("Ana", 34), ("Bo", 17)
add to users (name) the rows ("Ana"), ("Bo")
add to archive (name) get name from users where age is over 90
add to users default values
add or replace to users: id = 1, name = "Ana"      # also: or ignore, or abort, or fail
add to users: id = 1, name = "A" on conflict (id) do update set name = excluded.name
add to users: id = 1 on conflict do nothing
update users set age = 35 and city = "Rome" where name is "Ana"
change users set age to age + 1
delete from users where age is under 18
remove from users where id = 1 returning name
delete everything from users
```

`update` and `delete` without a `where` affect every row, just as they do in SQL.

## Changing structure

```text
create table users with id integer primary key auto increment, name string required, age integer default 18
create table if not exists tags (item integer, tag text, primary key (item, tag)) without rowid
create table pets (id integer primary key, owner integer references users(id) on delete cascade, check (id > 0))
create table recent as get * from users where age is under 30
create index idx_age on users (age desc)
create unique index if not exists idx_name on users (name) where name has a value
create view adults as get name from users where age is at least 18
create trigger log_new after insert on users for each row do (add to log: msg = new.name; delete from log where msg is null)
add column note text to users
drop column note from users
rename column note to memo in users
rename table users to people
drop table if exists users           # also: drop view, drop index, drop trigger
```

Column options: `primary key`, `auto increment`, `required` or `not null`, `unique`, `default x` or `defaulting to x`, `check (...)`, `references t(c)`, `collate x`, `generated always as (...)`. The types `string` and `number` become `TEXT` and `NUMERIC`. Any other type is passed through in uppercase.

## SQLite commands

| Write | Output |
|---|---|
| `begin transaction`, `start transaction`, `begin immediate` | `BEGIN ...;` |
| `commit`, `rollback` | `COMMIT;`, `ROLLBACK;` |
| `savepoint s`, `release s`, `rollback to s` | same, as SQL |
| `explain ...`, `explain query plan ...` | `EXPLAIN ...` |
| `pragma foreign_keys = on`, `pragma table_info(users)` | `PRAGMA ...;` |
| `vacuum`, `vacuum into "f.db"`, `analyze`, `reindex` | same, as SQL |
| `attach "f.db" as other`, `detach other` | `ATTACH ...`, `DETACH ...` |
| `show columns of users` | `PRAGMA table_info(users);` |
| `show tables`, `list tables` | `.tables` |
| `show schema`, `show schema of users`, `describe users` | `.schema`, `.schema users` |
| `show indexes`, `show indexes on users` | `.indexes`, `.indexes users` |
| `show databases` | `.databases` |
| `dump database`, `dump table users` | `.dump`, `.dump users` |
| `import "users.csv" into users` | `.import --csv users.csv users` |
| `run file "setup.sql"` | `.read setup.sql` |
| `backup to "copy.db"` | `.backup copy.db` |

## Anything else

Two escape hatches cover what the format doesn't have words for:

- A line starting with `.` is a `sqlite3` shell command and is passed through unchanged: `.mode column`.
- A line starting with `raw:` is SQL and is passed through unchanged: `raw: CREATE VIRTUAL TABLE docs USING fts5(body)`.

## What it does not check

The translator doesn't read your database, so it can't tell whether a table or column exists. With `--db`, SQLite reports those errors when the command runs. Some syntax also depends on your SQLite version: `RIGHT`/`FULL JOIN` and `IS DISTINCT FROM` need 3.39, and `RETURNING` needs 3.35.
