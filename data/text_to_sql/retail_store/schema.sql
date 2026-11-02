-- Schema for retail_store.sqlite (10 tables, 50 rows each)
CREATE TABLE stores (
    store_id      INTEGER PRIMARY KEY,
    store_name    TEXT NOT NULL,
    city          TEXT NOT NULL,
    state         TEXT NOT NULL,
    opened_date   TEXT NOT NULL            -- YYYY-MM-DD
);

CREATE TABLE employees (
    employee_id   INTEGER PRIMARY KEY,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    title         TEXT NOT NULL,
    store_id      INTEGER NOT NULL REFERENCES stores(store_id),
    manager_id    INTEGER REFERENCES employees(employee_id),
    hire_date     TEXT NOT NULL,           -- YYYY-MM-DD
    salary        INTEGER NOT NULL         -- yearly, in US dollars
);

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    email         TEXT NOT NULL,
    city          TEXT NOT NULL,           -- free text, inconsistent spellings
    state         TEXT NOT NULL,
    signup_date   TEXT NOT NULL            -- YYYY-MM-DD
);

CREATE TABLE suppliers (
    supplier_id   INTEGER PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    country       TEXT NOT NULL,
    contact_email TEXT NOT NULL
);

CREATE TABLE products (
    product_id        INTEGER PRIMARY KEY,
    product_name      TEXT NOT NULL,
    category          TEXT NOT NULL,
    supplier_id       INTEGER NOT NULL REFERENCES suppliers(supplier_id),
    unit_price_cents  INTEGER NOT NULL     -- price in CENTS, not dollars
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    store_id      INTEGER NOT NULL REFERENCES stores(store_id),
    employee_id   INTEGER NOT NULL REFERENCES employees(employee_id),
    order_date    TEXT NOT NULL,           -- stored as MM/DD/YYYY text
    status        TEXT NOT NULL            -- delivered, shipped, processing, cancelled
);

CREATE TABLE order_items (
    order_item_id     INTEGER PRIMARY KEY,
    order_id          INTEGER NOT NULL REFERENCES orders(order_id),
    product_id        INTEGER NOT NULL REFERENCES products(product_id),
    quantity          INTEGER NOT NULL,
    unit_price_cents  INTEGER NOT NULL     -- price paid, in CENTS
);

CREATE TABLE payments (
    payment_id    INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    method        TEXT NOT NULL,           -- credit_card, debit_card, paypal, gift_card
    amount_cents  INTEGER NOT NULL,
    status        TEXT NOT NULL            -- completed, refunded
);

CREATE TABLE shipments (
    shipment_id     INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(order_id),
    carrier         TEXT NOT NULL,
    shipped_date    TEXT,                  -- YYYY-MM-DD, NULL if not shipped
    delivered_date  TEXT,                  -- YYYY-MM-DD, NULL if not delivered
    status          TEXT NOT NULL          -- delivered, in_transit, pending, cancelled
);

CREATE TABLE reviews (
    review_id     INTEGER PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    rating        INTEGER NOT NULL,        -- 1 to 5
    review_text   TEXT NOT NULL,
    review_date   TEXT NOT NULL            -- YYYY-MM-DD
);
