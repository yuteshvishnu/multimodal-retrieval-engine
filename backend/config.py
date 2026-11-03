"""
All API keys are read from the .env file at the project root.
See .env.example for the full list and where each key is used.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

# also puts the keys in os.environ, for libraries that read them directly
load_dotenv(ROOT_DIR / ".env")

# Groq: RAG answers and Text-to-SQL
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq models (see what your key can use: https://console.groq.com/docs/models)
GROQ_MODEL_RAG = "openai/gpt-oss-20b"
GROQ_MODEL_SQL = "openai/gpt-oss-120b"
GROQ_MODEL_SUMMARY = "openai/gpt-oss-120b"  # offline table summaries (Text-to-SQL Version-3)

# Text-to-SQL Version-4: SQL is generated in rounds of 3 calls and compared; never more than this per question
SQL_CALLS_PER_ROUND = 3
MAX_LLM_CALLS_PER_QUESTION = 6
SQL_TEMPERATURE = 0.7  # some variety, so agreeing answers actually mean something

# Scout Version-8: performance limits for heavy queries
MAX_RESULT_ROWS = 200            # rows kept per query; fetch one more to know if there are more
PAGE_SIZE = 10                   # rows per page in the UI
QUERY_TIMEOUT_SECONDS = 5        # a query running longer than this is stopped
MAX_ESTIMATED_READS = 1_000_000  # rows a query may read, estimated from its plan before running
RESULT_CACHE_SIZE = 10           # most recent results kept for paging (least recently used is dropped)
ANSWER_CACHE_SIZE = 10           # Version-12: recent answers reused when the same question comes back
QUERY_CONNECTIONS = 4            # queries that can run at the same time (one slow query doesn't block the rest)

# Text-to-SQL Version-5: the user confirms what the question means first; at most this many sets of options
MAX_CLARIFY_ROUNDS = 3
# Version-10: tries at the steps (the first + re-runs from the user's comments)
MAX_STEP_ROUNDS = 5

# Version-11: the Failure Analyst agent (backend/agent/). It runs only when the developer asks
# (Dashboard -> Issues), so a stronger model than Scout's can be tried here; it must support tool calling.
AGENT_MODEL = "openai/gpt-oss-120b"
AGENT_MAX_TOOL_CALLS = 8   # read-only tool calls per issue; then it must write its report
AGENT_BATCH = 5            # "Investigate all new" handles at most this many issues per click

# every LLM call (backend/llm.py): fail instead of hanging, and retry rate limits / server errors
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 3

# Dashboard: $ per million tokens (input, output), to price every LLM call. Reasoning tokens
# are billed as output. Groq on-demand prices as listed on 2026-11-03 (third-party trackers;
# Groq's console is the source of truth: update these if they differ).
LLM_PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}

# Privacy: Groq sees the question and the schema. When False, Scout never sends data rows:
# no sample rows when writing table descriptions, no result rows in the disagreement feedback.
# (Scribe always sends the passages it answers from: that's how RAG works.)
SHARE_ROWS_WITH_LLM = True

# Optional login: when set, the whole app asks for this password (any username) before
# anything loads. Empty = no login, fine while the app only runs on your own machine.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# Hugging Face: RAG evaluation scripts
HF_TOKEN = os.getenv("HF_TOKEN", "")
