# Scout on BIRD: Findings so far, and Version-3 (Binding)

Yutesh, November 8, 2026. Scout is the text-to-SQL part of Rexie. This page sums up every BIRD experiment so far, what we learned, and why the next version is about **binding**.

## The short version

- **Best results:** Adaptive got **225 of 251 (89.6%)** on corrected BIRD questions 50–300, against Original's 217 (86.5%), at about 28% lower cost. On corrected 1–50, Adaptive with the new semantic and value checks got **47 of 50**.
- **Voting isn't the main limit.** In 72 of 86 misses studied, **no candidate answer was right**, so no better vote could have fixed them. The model writes wrong SQL, and all its tries make the same mistake.
- **Most of those mistakes are binding errors.** Scout understands what the question asks, but attaches its pieces to the wrong place in the data: a number to the wrong column, a word to the wrong field, an average to the wrong denominator, a "month" to a whole date.
- **Code checks fixed what prompt rules couldn't.** Rules in prompts were often ignored. Checking the SQL against the data after it ran fixed #1524 and #1526 on the first live try.
- **Version-3 makes binding its own step before the SQL is written,** so the mistake isn't made in the first place instead of being caught and retried.
- **Nothing here is a final score.** All work used questions 1–300. Questions 301–500 are untouched, for one clean test.

## Every experiment, side by side

| Experiment | What we tried | Result | Verdict |
|---|---|---|---|
| Exact full-row scoring | Score every row, exactly as BIRD's own scorer | Removed false misses (the old scorer read only 200 rows); 0 of 442 rights changed by exact comparison | Essential |
| Corrected answer key | Score the same SQL on BIRD's official and the corrected key (UIUC Kang lab) | 18 of 100 results flip between keys | Essential |
| Original | 1 meaning, a plan, 3 SQL tries, a retry round, majority vote (5–6 calls) | 217/251 on corrected 50–300 | Baseline |
| Full (Ask Me Anything) | 4 chains asking the same task in different ways, majority vote (8 calls) | 47 vs Original 43 on 50–100; 40 vs 42 on 1–50 (older design) | Unclear |
| Weak supervision | Weigh votes by accuracy learned only from agreement (triplet method) | 46 vs 47, and 75 vs 76: one below a plain vote both times | Didn't help |
| Adaptive | 2 chains; stop if they agree with data, else 4; one retry (4–12 calls) | 225 vs 217 of 251; 4.3 vs 5.1 calls; 28% cheaper | Helped (cost, speed) |
| Mixed | 3 models from 3 labs (GPT-OSS 120B, DeepSeek V4.1 Flash, Nemotron 3.5 Super) | 76/100 official, equal to DeepSeek alone; all 3 wrong on 20 | Not proven, most expensive |
| SQLite refresher | ~60 SQL examples (3,700 tokens) in every SQL call | 24 vs 25 of 35; 0 fixed, 1 broken; tokens nearly doubled | Didn't help, off |
| English program | Model writes strict English, `english2sqlite.py` compiles it to SQL | 22 → 24 of 30 after 2 format changes; no translation mistakes found | Promising, unproven |
| Data beats empty | An empty answer wins a vote only if no chain found data | Replay: 4 fixed, 1 broken (an empty official key) | Small help |
| Prompt rules | "Respectively", totals before extremes, which metric, date parts, average denominators, value grounding | #1480 and #1498 stayed wrong after their rules were added | Often ignored |
| Semantic checks (code) | Meaning returns a spec (measure, grain, extreme, date part, values per row); code checks SQL and rows against it, then retries | Replay on 2,226 saved answers: 39 wrong flagged, 0 of 1,834 right flagged | Helped |
| Value check (code) | `VALUE_GROUNDING`: no rows, and the question's number lives in another column | Replay: flagged #1524/#1526 only; live run fixed both | Helped |

## Results by run

Compare scores only within a row: each row is a different set of questions.

| Questions and key | Original | Full | Full + WS | Adaptive | Mixed | Mixed + WS | English |
|---|---|---|---|---|---|---|---|
| Corrected 1–50 (older design) | 42 | 40 | | 41 | | | |
| Corrected 50–100 | 43 | 47 | 46 | 47 (replay) | | | |
| Corrected 101–200 | 92 | | | 90 | | | |
| Corrected 200–300 | 82 | | | 88 | | | |
| **Corrected 50–300 (251)** | **217** | | | **225** | | | |
| Official 1–100 | | | | | 76 | 75 | |
| Corrected 1–30 | 24 | 23 | | 24 → 25 (new rules) | | | 24 |
| **Corrected 1–50, with checks** | | | | 46 → **47** | | | |

The last row's two runs: 46/50 before the value check, 47/50 after it (run `20261108-183333-1143`), at $0.066 for 50 questions.

## Where the mistakes are

**Was a right answer there to pick?**

| Flow and range | Misses | A right option was present |
|---|---|---|
| Original, corrected 50–300 | 35 | 3 |
| Full, corrected 50–100 | 4 | 2 |
| Adaptive, corrected 101–300 | 23 | 5 |
| Mixed, official 1–100 | 24 | 4 |
| **Total** | **86** | **14** |

So better voting, weak supervision or more models could fix at most 14 of 86. The other 72 need better SQL.

**The remaining misses on corrected 1–50 (Adaptive, latest runs):**

| Question | What went wrong | Kind |
|---|---|---|
| #1524 | Matched 548.4 against `Amount × Price`; the data stores it in `Price` | Value → column (fixed by the value check) |
| #1526 | Matched 634.8 against `Amount × Price`; it's in `Price` | Value → column (fixed by the value check) |
| #1323 | Looked for "fundraiser" in `notes` instead of the `type` column; one chain was right but lost | Word → column |
| #1480 | Returned `201304` when "which month" wants `04` | Question part → date part (fixed by a check) |
| #1498 | Highest single record instead of the highest monthly total | Measure → grain (fixed by a check) |
| #1339 | Averaged per event; the key divides by budget lines | Average → denominator (doubtful key) |
| #1486 | Computed money; the key counts customers, though the question says "pay" | Measure (doubtful key) |

## Why this is a binding problem

A question has pieces: numbers ("548.4"), names ("fundraiser"), measures ("how much", "how many"), units ("per event", "monthly") and output parts ("which month"). Writing SQL means **binding** each piece to something in the database: a column, an expression, a table, a grouping. Our misses show Scout usually **understands** each piece but **binds** it wrongly:

- "spent 548.4" was understood as money, and bound to `Amount × Price` instead of `Price`.
- "not fundraisers" was understood correctly, and bound to `notes` instead of `type`.
- "which month" was understood as a month, and bound to the whole `YYYYMM` value.
- "highest monthly consumption" was understood, and bound to single rows instead of monthly totals.

This matches the failure Sala's group reports in **"Models Can Model, But Can't Bind: Structured Grounding in Text-to-Optimization"** (Gao, Ge, Berenbeim, Bastian, Sala; COLM 2026): models build the right overall model of a problem but fail to bind its parts to the right data. Our results suggest the same happens in text-to-SQL, and they explain the other findings:

- **Voting can't fix it:** every chain binds from the same schema and the same guess, so they agree on the same wrong binding (72 of 86 misses had no right option).
- **More models don't fix it:** three labs' models made the same binding mistakes (all wrong on 20 of 100).
- **More SQL knowledge doesn't fix it:** the refresher taught commands, not where values live (0 fixed).
- **Checks against the data do fix it:** #1524 and #1526 were fixed only when code looked in the data and said where 548.4 actually is.

## Version-3: Binding before SQL

**Now:** question → meaning → model writes SQL → run → checks flag a mistake → retry.

**Version-3:** question → meaning → **binding** → model writes SQL from the confirmed bindings → run → checks.

The binding step lists each piece of the question and binds it, checked by code against the data wherever possible:

```
"spent 548.4 on 2012/8/24"    (#1524)
  548.4      model's guess: money spent (Amount × Price)
             data search: found only in transactions_1k.Price (1 row)        bind to Price
  2012/8/24  data search: transactions_1k.Date = '2012-08-24'                bound
  "segment"  customers.Segment                                                bound
```

The SQL step then only has to join and select what's bound. The same data lookup that today runs after an empty answer runs first, so every chain starts from the right binding instead of all guessing the same wrong one.

**What gets bound, and how it's checked:**

| Piece of the question | Bound to | Checked by code |
|---|---|---|
| A number or name ("548.4", "fundraiser") | The column(s) that hold it | Search the data's values; prefer a stored column over a calculation |
| A measure ("how much", "how many") | A count, a sum of money, or another metric | The measure check on the SQL's totals |
| A unit ("per event", "monthly") | The grouping key and the denominator | Does the unit repeat in the table? Do two denominators differ? |
| An output part ("which month") | The part of the field returned | Rows checked for whole dates |
| A lookup ("the customer who…") | One record | Does the condition match exactly one record? (new) |

**Why test it, and how:**

1. **The question:** does binding first beat checking after? Version-3 runs on the same questions as the current Adaptive with checks, with the same model and settings, so only the binding step changes.
2. **Measures:** right of total, how many misses had a wrong binding, how many bindings the data confirmed, and calls and cost per question.
3. **Fairness:** binding uses only the question, its hint, the schema and the data, never the answer key. No rule names a question or database.
4. **Order:** develop on corrected 1–300 (already studied), set aside doubtful keys (#1486, #1339) when comparing, then **one clean run on 301–500**, which no one has studied.

**What would show it works:** fewer misses with no right option, fewer retries (mistakes avoided, not repaired) and a gain on 301–500, not only on studied questions.

## Not done yet

- A check that a `LIMIT 1` lookup matches exactly one record (`ENTITY_LOOKUP_AMBIGUITY`).
- BIRD's own scoring script hasn't been run on Scout's answers.
- The English program hasn't been run on 50–300.
- No fair Original run on official 1–100.
- The clean test on 301–500.
