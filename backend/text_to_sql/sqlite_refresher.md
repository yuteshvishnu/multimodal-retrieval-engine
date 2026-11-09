# SQLite refresher

Reference only: what each command does, with examples. Use a command only when the question needs it. The examples use a made-up shop, so never copy its table or column names: use only the Schema you are given. They show how commands work, not which columns to return; the instructions above decide that.

Made-up shop:
customers(id, name, city, phone, signup_date)
products(id, name, category, price)
orders(id, customer_id, order_date, status, total)
order_items(order_id, product_id, qty, unit_price)

Only one read query can be run: SELECT, or WITH ... SELECT.

## Reading rows

SELECT picks columns, FROM picks the table, AS names a column or table.
```sql
SELECT name AS customer_name, city FROM customers
```

WHERE keeps rows that meet a condition. AND, OR, NOT combine conditions; AND runs before OR, so use brackets.
```sql
SELECT id FROM orders WHERE status = 'paid' AND (total > 100 OR customer_id = 7)
```

ORDER BY sorts (ASC by default, or DESC). NULLS FIRST / NULLS LAST choose where empty values go. LIMIT keeps the first n rows; OFFSET skips rows first.
```sql
SELECT name, price FROM products ORDER BY price DESC NULLS LAST, name ASC LIMIT 5 OFFSET 10
```

DISTINCT removes duplicate result rows.
```sql
SELECT DISTINCT city FROM customers
```

## Conditions

Comparisons: =, <>, !=, <, <=, >, >=. BETWEEN includes both ends.
```sql
SELECT id FROM orders WHERE total BETWEEN 50 AND 100
```

IN matches any value in a list or subquery.
```sql
SELECT name FROM products WHERE category IN ('Toys', 'Books')
```

LIKE matches a pattern: % is any text, _ is one character. It ignores case for ASCII letters. = does not ignore case. GLOB is like LIKE, but uses * and ?, and respects case.
```sql
SELECT name FROM customers WHERE name LIKE 'an%' OR name GLOB 'J?n*'
```

NULL means empty or unknown. NULL never equals anything, even NULL, so use IS NULL or IS NOT NULL.
```sql
SELECT name FROM customers WHERE phone IS NULL
```

EXISTS is true when a subquery returns any row. NOT EXISTS is true when it returns none.
```sql
SELECT c.name FROM customers AS c WHERE EXISTS (SELECT 1 FROM orders AS o WHERE o.customer_id = c.id)
```

## Joins

JOIN (same as INNER JOIN) keeps only rows that match on both sides.
```sql
SELECT o.id, c.name FROM orders AS o JOIN customers AS c ON o.customer_id = c.id
```

LEFT JOIN keeps every row of the left table; columns from the right table are NULL when nothing matches. A WHERE condition on the right table removes those rows; put the condition in ON to keep them.
```sql
SELECT c.name, o.id FROM customers AS c LEFT JOIN orders AS o ON o.customer_id = c.id AND o.status = 'paid'
```

RIGHT JOIN keeps every row of the right table. FULL JOIN keeps every row of both.
```sql
SELECT c.name, o.id FROM customers AS c RIGHT JOIN orders AS o ON o.customer_id = c.id
```
```sql
SELECT c.name, o.id FROM customers AS c FULL JOIN orders AS o ON o.customer_id = c.id
```

CROSS JOIN pairs every row with every row.
```sql
SELECT c.city, p.category FROM customers AS c CROSS JOIN products AS p
```

USING joins on a column with the same name in both tables. A self join uses one table twice, with different names.
```sql
SELECT i.qty, j.qty FROM order_items AS i JOIN order_items AS j USING (order_id) WHERE i.product_id < j.product_id
```

A join repeats a row once per match, so totals and counts over a join can count the same thing more than once. COUNT(DISTINCT ...) or a subquery avoids that.

## Totals and groups

COUNT(*) counts rows; COUNT(col) counts non-NULL values; COUNT(DISTINCT col) counts different non-NULL values. SUM, AVG, MIN and MAX skip NULLs. TOTAL is like SUM but gives 0.0 instead of NULL when there is nothing to add.
```sql
SELECT COUNT(*), COUNT(phone), COUNT(DISTINCT city) FROM customers
```
```sql
SELECT SUM(total), TOTAL(total), AVG(total), MIN(total), MAX(total) FROM orders
```

GROUP_CONCAT joins the values of a group into one text. STRING_AGG does the same and needs a separator. MEDIAN gives the middle value.
```sql
SELECT category, GROUP_CONCAT(name, ', '), STRING_AGG(name, '; '), MEDIAN(price) FROM products GROUP BY category
```

GROUP BY gives one result row per group. WHERE filters rows before grouping; HAVING filters groups after.
```sql
SELECT customer_id, SUM(total) AS spent FROM orders WHERE status = 'paid' GROUP BY customer_id HAVING SUM(total) > 500
```

FILTER limits one total to some rows.
```sql
SELECT COUNT(*) FILTER (WHERE status = 'paid'), COUNT(*) FILTER (WHERE status = 'open') FROM orders
```

## Choosing a value

CASE gives a value by condition. Without ELSE, a row that matches nothing gets NULL. IIF(condition, a, b) is a short CASE.
```sql
SELECT id, CASE WHEN total >= 100 THEN 'large' WHEN total >= 50 THEN 'medium' ELSE 'small' END FROM orders
```
```sql
SELECT id, IIF(status = 'paid', total, 0) FROM orders
```

COALESCE gives the first non-NULL value; IFNULL does the same for two values. NULLIF(a, b) gives NULL when a equals b, which avoids dividing by zero.
```sql
SELECT name, COALESCE(phone, city, 'none'), IFNULL(phone, 'none') FROM customers
```
```sql
SELECT SUM(total) / NULLIF(COUNT(*), 0) FROM orders
```

## Numbers

Dividing two whole numbers gives a whole number: 5 / 2 is 2. Make one of them decimal for 2.5. CAST converts a value's type (INTEGER, REAL, TEXT).
```sql
SELECT 5 / 2, 5 * 1.0 / 2, CAST(5 AS REAL) / 2, CAST('12' AS INTEGER)
```

Arithmetic: + - * / and % (remainder). ROUND(x, n) rounds, ABS drops the sign, and MAX(a, b) and MIN(a, b) with two or more values pick between them on one row. CEIL, FLOOR, POWER, SQRT and LOG also work.
```sql
SELECT ROUND(price, 2), ABS(price - 10), MAX(price, 5), MIN(price, 5), price % 2, CEIL(price), FLOOR(price), POWER(price, 2), SQRT(ABS(price)), LOG(ABS(price) + 1) FROM products
```

Text that holds numbers sorts as text ('9' comes after '10'), so CAST it to sort or compare it as a number.
```sql
SELECT id FROM orders ORDER BY CAST(total AS REAL) DESC
```

## Text

|| joins text. CONCAT and CONCAT_WS also join text, skipping NULLs. UPPER, LOWER, TRIM, LTRIM and RTRIM change case and spaces. LENGTH counts characters.
```sql
SELECT name || ' (' || city || ')', CONCAT(name, city), CONCAT_WS('-', name, city), UPPER(name), LOWER(name), TRIM(name), LTRIM(name), RTRIM(name), LENGTH(name) FROM customers
```

SUBSTR(text, start, length) cuts text; positions start at 1 and a negative start counts from the end. INSTR finds a position (0 if not found). REPLACE swaps text. PRINTF formats values.
```sql
SELECT SUBSTR(phone, 1, 3), SUBSTR(phone, -4), INSTR(name, ' '), REPLACE(phone, '-', ''), PRINTF('%.2f', 3.14159) FROM customers
```

The part of a text before and after a separator:
```sql
SELECT SUBSTR(name, 1, INSTR(name, ' ') - 1) AS first_part, SUBSTR(name, INSTR(name, ' ') + 1) AS last_part FROM customers WHERE INSTR(name, ' ') > 0
```

TYPEOF tells a value's stored type. It helps when a column holds mixed values.
```sql
SELECT TYPEOF(total), COUNT(*) FROM orders GROUP BY TYPEOF(total)
```

## Dates

SQLite has no date type. Dates are usually text like '2024-03-15', and these functions need that form. There is no YEAR(), MONTH(), DATEDIFF() or EXTRACT().

STRFTIME(format, date) takes parts of a date, as text: %Y year, %m month, %d day, %H hour, %M minute, %S second, %w weekday (0 is Sunday), %j day of the year.
```sql
SELECT id FROM orders WHERE STRFTIME('%Y', order_date) = '2024' AND STRFTIME('%m', order_date) = '03'
```

DATE, TIME and DATETIME give a date or time, and can move it. 'now' is the current time.
```sql
SELECT DATE('2024-03-15', '+7 days'), DATE('2024-03-15', '-1 month'), DATE('2024-03-15', 'start of month'), TIME('2024-03-15 10:30:00'), DATETIME('2024-03-15', '+1 hour'), DATE('now')
```

JULIANDAY gives a date as a number, so subtracting gives the days between. UNIXEPOCH gives seconds since 1970. TIMEDIFF gives the difference as text.
```sql
SELECT JULIANDAY('2024-03-15') - JULIANDAY('2024-03-01'), UNIXEPOCH('2024-03-15'), TIMEDIFF('2024-03-15', '2024-03-01')
```

Age in whole years, on a given date:
```sql
SELECT name, STRFTIME('%Y', '2024-03-15') - STRFTIME('%Y', signup_date) - (STRFTIME('%m-%d', '2024-03-15') < STRFTIME('%m-%d', signup_date)) AS years FROM customers
```

## Subqueries and WITH

A subquery can give one value, a list, or a table to read from.
```sql
SELECT name FROM products WHERE price = (SELECT MAX(price) FROM products)
```
```sql
SELECT name FROM customers WHERE id NOT IN (SELECT customer_id FROM orders WHERE customer_id IS NOT NULL)
```
```sql
SELECT AVG(spent) FROM (SELECT customer_id, SUM(total) AS spent FROM orders GROUP BY customer_id)
```

A correlated subquery runs once per outer row and can use its columns.
```sql
SELECT p.name FROM products AS p WHERE p.price > (SELECT AVG(q.price) FROM products AS q WHERE q.category = p.category)
```

WITH names a query to use below, which keeps long queries readable.
```sql
WITH spending AS (SELECT customer_id, SUM(total) AS spent FROM orders GROUP BY customer_id)
SELECT c.name, s.spent FROM spending AS s JOIN customers AS c ON c.id = s.customer_id WHERE s.spent > 500
```

WITH RECURSIVE builds rows step by step, such as a list of months.
```sql
WITH RECURSIVE months(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM months WHERE n < 12)
SELECT n FROM months
```

## Combining results

UNION joins results and removes duplicates; UNION ALL keeps them. INTERSECT keeps rows in both; EXCEPT keeps rows in the first but not the second. Both sides need the same number of columns.
```sql
SELECT city FROM customers UNION SELECT category FROM products
```
```sql
SELECT customer_id FROM orders WHERE status = 'paid' UNION ALL SELECT customer_id FROM orders WHERE status = 'open'
```
```sql
SELECT customer_id FROM orders WHERE total > 100 INTERSECT SELECT customer_id FROM orders WHERE status = 'paid'
```
```sql
SELECT id FROM customers EXCEPT SELECT customer_id FROM orders
```

## Window functions

These work per row while seeing other rows; they don't merge rows as GROUP BY does. OVER sets the rows they see, PARTITION BY splits them into groups, and ORDER BY sets their order.

ROW_NUMBER numbers rows with no repeats. RANK gives equal values the same rank and skips the next one (1, 1, 3). DENSE_RANK doesn't skip (1, 1, 2).
```sql
SELECT name, ROW_NUMBER() OVER (ORDER BY price DESC), RANK() OVER (ORDER BY price DESC), DENSE_RANK() OVER (ORDER BY price DESC) FROM products
```

LAG and LEAD read the previous and next row. FIRST_VALUE, LAST_VALUE and NTH_VALUE read a chosen row. NTILE splits rows into equal groups; PERCENT_RANK and CUME_DIST give a relative position.
```sql
SELECT id, total, LAG(total) OVER (ORDER BY order_date), LEAD(total) OVER (ORDER BY order_date), FIRST_VALUE(total) OVER (ORDER BY order_date), NTH_VALUE(total, 2) OVER (ORDER BY order_date), NTILE(4) OVER (ORDER BY total), PERCENT_RANK() OVER (ORDER BY total), CUME_DIST() OVER (ORDER BY total) FROM orders
```

Totals also work as window functions. A frame sets which rows count.
```sql
SELECT id, SUM(total) OVER (ORDER BY order_date), AVG(total) OVER (PARTITION BY customer_id), AVG(total) OVER (ORDER BY order_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM orders
```

## JSON

JSON_EXTRACT reads a value from JSON text; -> and ->> are short forms.
```sql
SELECT JSON_EXTRACT('{"size": 3}', '$.size'), '{"size": 3}' ->> '$.size', JSON_ARRAY_LENGTH('[1, 2, 3]')
```

## Writing names and values

Text values use single quotes; an apostrophe inside is written twice. Names with spaces or odd characters use double quotes or backticks.
```sql
SELECT 'O''Brien' AS "customer name", `name` FROM customers WHERE "city" = 'St. John''s'
```

## Combined examples

Every row tied for the highest total, per group or overall:
```sql
WITH spending AS (SELECT customer_id, SUM(total) AS spent, RANK() OVER (ORDER BY SUM(total) DESC) AS place FROM orders GROUP BY customer_id)
SELECT c.name, s.spent FROM spending AS s JOIN customers AS c ON c.id = s.customer_id WHERE s.place = 1
```
```sql
SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) = (SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM orders GROUP BY customer_id))
```

The top 2 products in each category:
```sql
SELECT category, name FROM (SELECT category, name, ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS place FROM products) WHERE place <= 2
```

Each customer's most recent order:
```sql
SELECT id, customer_id, order_date FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS place FROM orders) WHERE place = 1
```

Customers with no orders, including those with no match at all:
```sql
SELECT c.name FROM customers AS c LEFT JOIN orders AS o ON o.customer_id = c.id WHERE o.id IS NULL
```

The percentage of orders that are paid, and of customers with at least one paid order:
```sql
SELECT SUM(CASE WHEN status = 'paid' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), COUNT(DISTINCT CASE WHEN status = 'paid' THEN customer_id END) * 100.0 / COUNT(DISTINCT customer_id) FROM orders
```

Turning rows into columns:
```sql
SELECT customer_id, SUM(CASE WHEN STRFTIME('%Y', order_date) = '2023' THEN total ELSE 0 END) AS y2023, SUM(CASE WHEN STRFTIME('%Y', order_date) = '2024' THEN total ELSE 0 END) AS y2024 FROM orders GROUP BY customer_id
```

The total for each month, and the month with the highest total:
```sql
SELECT STRFTIME('%Y-%m', order_date) AS month, SUM(total) AS month_total FROM orders GROUP BY month ORDER BY month_total DESC LIMIT 1
```

Change from the previous month, with a running total:
```sql
WITH monthly AS (SELECT STRFTIME('%Y-%m', order_date) AS month, SUM(total) AS month_total FROM orders GROUP BY month)
SELECT month, month_total - LAG(month_total) OVER (ORDER BY month), (month_total - LAG(month_total) OVER (ORDER BY month)) * 100.0 / LAG(month_total) OVER (ORDER BY month), SUM(month_total) OVER (ORDER BY month) FROM monthly
```

A total of quantity times price per category, across several tables:
```sql
SELECT p.category, SUM(i.qty * i.unit_price) AS revenue, COUNT(DISTINCT o.customer_id) AS buyers FROM order_items AS i JOIN products AS p ON p.id = i.product_id JOIN orders AS o ON o.id = i.order_id WHERE o.status = 'paid' GROUP BY p.category HAVING COUNT(DISTINCT o.customer_id) >= 2 ORDER BY revenue DESC
```

Customers who bought in both of two categories:
```sql
SELECT o.customer_id FROM orders AS o JOIN order_items AS i ON i.order_id = o.id JOIN products AS p ON p.id = i.product_id WHERE p.category IN ('Toys', 'Books') GROUP BY o.customer_id HAVING COUNT(DISTINCT p.category) = 2
```

The average, leaving out placeholder values:
```sql
SELECT AVG(NULLIF(price, 0)) FROM products
```

End of the reference.
