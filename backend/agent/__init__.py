"""
Version-11: the Failure Analyst, an AI agent that investigates Scout's failures (wrong eval
answers, users' reports, SQL errors) with read-only tools and reports what happened, why,
a suggested fix and a test of that fix. The developer approves or rejects every fix.

    tools.py    what the agent may do: look at tables and run small read-only queries
    analyst.py  the agent loop: LLM -> tool call -> result -> LLM ... -> report
    fixes.py    tests a suggested fix (code, not the LLM), and applies it once approved
"""
