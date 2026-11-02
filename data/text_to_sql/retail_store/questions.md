# Text-to-SQL Practice Set: 20 Questions

Database: `retail_store.sqlite` (10 tables, 50 rows each). Schema: `schema.sql`.

Use Part 1 to test your system. Check its answers against Part 2, the answer key.

## Part 1: Questions

### Easy

1. How many customers are there?
2. List the names of all products in the Electronics category.
3. What is the most expensive product, and what is its price in dollars?
4. How many orders have been delivered?
5. Which store has the most employees?
6. Which payment method is used most often?

### Medium

7. How many customers live in New York City?
8. How many orders were placed in March 2026?
9. Who are the top 5 customers by total amount spent, in dollars?
10. What is the total revenue in dollars for each store that has sales?
11. Which supplier provides the most products?
12. What is the average review rating for each product category?
13. Which products have never been reviewed?
14. What is the average delivery time in days for delivered shipments?

### Hard

15. How many unique customers placed at least one order in the second quarter (April to June) of 2026?
16. Which employees earn more than their manager?
17. Which customers have placed orders but never written a review?
18. What percentage of all orders were cancelled?
19. Which product category generated the most revenue, and how much in dollars?
20. For each store that has orders, which employee handled the most orders there?

## Part 2: Answer key

Each answer below came from running the correct SQL on the database. Questions with a **Trap** note are designed to catch common LLM mistakes.

### Q1. How many customers are there?
*Difficulty: easy*

```sql
SELECT COUNT(*) FROM customers;
```

Answer: **50**

### Q2. List the names of all products in the Electronics category.
*Difficulty: easy*

```sql
SELECT product_name FROM products WHERE category = 'Electronics' ORDER BY product_name;
```

Answer:

| product_name |
|---|
| 4K Monitor |
| Bluetooth Speaker |
| Mechanical Keyboard |
| Noise Cancelling Headphones |
| Portable SSD 1TB |
| Smart Watch |
| USB-C Charger |
| Webcam HD |
| Wireless Earbuds |
| Wireless Mouse |

### Q3. What is the most expensive product, and what is its price in dollars?
*Difficulty: easy*

**Trap:** Prices are stored in cents. A model that returns 39999 as the price in dollars is wrong.

```sql
SELECT product_name, unit_price_cents / 100.0 AS price_dollars FROM products ORDER BY unit_price_cents DESC LIMIT 1;
```

Answer:

| product_name | price_dollars |
|---|---|
| Stand Mixer | 399.99 |

### Q4. How many orders have been delivered?
*Difficulty: easy*

```sql
SELECT COUNT(*) FROM orders WHERE status = 'delivered';
```

Answer: **30**

### Q5. Which store has the most employees?
*Difficulty: easy*

```sql
SELECT s.store_name, COUNT(*) AS employee_count FROM employees e JOIN stores s ON e.store_id = s.store_id GROUP BY s.store_id ORDER BY employee_count DESC LIMIT 1;
```

Answer:

| store_name | employee_count |
|---|---|
| Brooklyn Main | 9 |

### Q6. Which payment method is used most often?
*Difficulty: easy*

```sql
SELECT method, COUNT(*) AS uses FROM payments GROUP BY method ORDER BY uses DESC LIMIT 1;
```

Answer:

| method | uses |
|---|---|
| debit_card | 19 |

### Q7. How many customers live in New York City?
*Difficulty: medium*

**Trap:** The city column has four spellings: 'New York', 'NY', 'NYC' and 'new york'. WHERE city = 'New York' undercounts.

```sql
SELECT COUNT(*) FROM customers WHERE LOWER(city) IN ('new york', 'ny', 'nyc');
```

Answer: **13**

### Q8. How many orders were placed in March 2026?
*Difficulty: medium*

**Trap:** order_date is TEXT in MM/DD/YYYY format, so a comparison like order_date >= '2026-03-01' silently fails.

```sql
SELECT COUNT(*) FROM orders WHERE order_date LIKE '03/%/2026';
```

Answer: **3**

### Q9. Who are the top 5 customers by total amount spent, in dollars?
*Difficulty: medium*

**Trap:** Needs a three-table join, cents-to-dollars conversion, and excluding cancelled orders.

```sql
SELECT c.first_name || ' ' || c.last_name AS customer, SUM(oi.quantity * oi.unit_price_cents) / 100.0 AS total_spent FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id WHERE o.status != 'cancelled' GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 5;
```

Answer:

| customer | total_spent |
|---|---|
| Thomas Patel | 1941.89 |
| Mary Williams | 1631.94 |
| Wei Baker | 859.9 |
| Ananya Kim | 677.93 |
| Karen Lee | 551.93 |

### Q10. What is the total revenue in dollars for each store that has sales?
*Difficulty: medium*

```sql
SELECT s.store_name, SUM(oi.quantity * oi.unit_price_cents) / 100.0 AS revenue FROM stores s JOIN orders o ON s.store_id = o.store_id JOIN order_items oi ON o.order_id = oi.order_id WHERE o.status != 'cancelled' GROUP BY s.store_id ORDER BY revenue DESC;
```

Answer:

| store_name | revenue |
|---|---|
| Madison Main | 2297.8 |
| New York Main | 2091.87 |
| Austin Main | 1429.76 |
| Milwaukee Main | 1293.87 |
| Seattle Main | 761.88 |
| Brooklyn Main | 689.81 |
| Houston Main | 604.89 |
| Chicago Main | 477.91 |
| San Francisco Main | 299.97 |
| Dallas Main | 139.94 |

### Q11. Which supplier provides the most products?
*Difficulty: medium*

```sql
SELECT su.supplier_name, COUNT(*) AS product_count FROM suppliers su JOIN products p ON su.supplier_id = p.supplier_id GROUP BY su.supplier_id ORDER BY product_count DESC LIMIT 1;
```

Answer:

| supplier_name | product_count |
|---|---|
| Blue Ridge Supply Co | 8 |

### Q12. What is the average review rating for each product category?
*Difficulty: medium*

```sql
SELECT p.category, ROUND(AVG(r.rating), 2) AS avg_rating FROM reviews r JOIN products p ON r.product_id = p.product_id GROUP BY p.category ORDER BY avg_rating DESC;
```

Answer:

| category | avg_rating |
|---|---|
| Clothing | 3.93 |
| Sports & Outdoors | 3.67 |
| Electronics | 3.64 |
| Home & Kitchen | 3.43 |
| Books | 3.22 |

### Q13. Which products have never been reviewed?
*Difficulty: medium*

**Trap:** Needs a LEFT JOIN (or NOT IN / NOT EXISTS). An inner join returns nothing useful.

```sql
SELECT p.product_name FROM products p LEFT JOIN reviews r ON p.product_id = r.product_id WHERE r.review_id IS NULL ORDER BY p.product_name;
```

Answer:

| product_name |
|---|
| 4K Monitor |
| Air Fryer |
| Bike Helmet |
| Blender |
| Bluetooth Speaker |
| Camping Tent |
| Cast Iron Skillet |
| Cotton T-Shirt |
| Cutting Board |
| Electric Kettle |
| Foam Roller |
| Hiking Backpack |

*...and 11 more rows (23 total).*

### Q14. What is the average delivery time in days for delivered shipments?
*Difficulty: medium*

```sql
SELECT ROUND(AVG(julianday(delivered_date) - julianday(shipped_date)), 2) AS avg_days FROM shipments WHERE status = 'delivered';
```

Answer: **3.63**

### Q15. How many unique customers placed at least one order in the second quarter (April to June) of 2026?
*Difficulty: hard*

**Trap:** Two traps: the text date format, and counting unique customers rather than orders.

```sql
SELECT COUNT(DISTINCT o.customer_id) FROM orders o WHERE substr(o.order_date,7,4)||'-'||substr(o.order_date,1,2)||'-'||substr(o.order_date,4,2) BETWEEN '2026-04-01' AND '2026-06-30';
```

Answer: **15**

### Q16. Which employees earn more than their manager?
*Difficulty: hard*

**Trap:** Needs a self-join on the employees table.

```sql
SELECT e.first_name || ' ' || e.last_name AS employee, e.salary, m.first_name || ' ' || m.last_name AS manager, m.salary AS manager_salary FROM employees e JOIN employees m ON e.manager_id = m.employee_id WHERE e.salary > m.salary;
```

Answer:

| employee | salary | manager | manager_salary |
|---|---|---|---|
| Sofia Mitchell | 82000 | Susan Singh | 78000 |
| Kenji Brown | 84500 | Carlos Patel | 83000 |

### Q17. Which customers have placed orders but never written a review?
*Difficulty: hard*

```sql
SELECT DISTINCT c.first_name || ' ' || c.last_name AS customer FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE c.customer_id NOT IN (SELECT customer_id FROM reviews) ORDER BY customer;
```

Answer:

| customer |
|---|
| Chloe Ali |
| Daniel Allen |
| Daniel Smith |
| Elizabeth Smith |
| Fatima Martinez |
| Jessica Allen |
| Karen Scott |
| Mei Chen |
| Michael Green |
| Ravi Nelson |
| Wei Baker |

### Q18. What percentage of all orders were cancelled?
*Difficulty: hard*

**Trap:** Integer division trap: without 100.0 the result rounds to 0.

```sql
SELECT ROUND(100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancelled_pct FROM orders;
```

Answer: **20.0**

### Q19. Which product category generated the most revenue, and how much in dollars?
*Difficulty: hard*

```sql
SELECT p.category, SUM(oi.quantity * oi.unit_price_cents) / 100.0 AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id JOIN orders o ON oi.order_id = o.order_id WHERE o.status != 'cancelled' GROUP BY p.category ORDER BY revenue DESC LIMIT 1;
```

Answer:

| category | revenue |
|---|---|
| Home & Kitchen | 4054.75 |

### Q20. For each store that has orders, which employee handled the most orders there?
*Difficulty: hard*

**Trap:** Top-1 per group. Ties are kept, so a store can list more than one employee.

```sql
WITH counts AS (SELECT o.store_id, o.employee_id, COUNT(*) AS n FROM orders o GROUP BY o.store_id, o.employee_id), ranked AS (SELECT *, RANK() OVER (PARTITION BY store_id ORDER BY n DESC) AS rk FROM counts) SELECT s.store_name, e.first_name || ' ' || e.last_name AS employee, r.n AS orders_handled FROM ranked r JOIN stores s ON r.store_id = s.store_id JOIN employees e ON r.employee_id = e.employee_id WHERE r.rk = 1 ORDER BY s.store_id, employee;
```

Answer:

| store_name | employee | orders_handled |
|---|---|---|
| New York Main | Arjun Tanaka | 2 |
| Brooklyn Main | Barbara Ali | 3 |
| Brooklyn Main | Ravi Khan | 3 |
| Chicago Main | Daniel Tanaka | 1 |
| Chicago Main | Elizabeth Hall | 1 |
| Madison Main | John Patel | 3 |
| Milwaukee Main | Aisha Patel | 1 |
| Milwaukee Main | Arjun Walker | 1 |
| Milwaukee Main | Michael Martinez | 1 |
| Milwaukee Main | Olivia Walker | 1 |
| Austin Main | Karen Mitchell | 4 |
| Austin Main | Richard Martinez | 4 |

*...and 4 more rows (16 total).*
