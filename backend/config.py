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

# Which company runs the LLM (Version-16). Both speak the same OpenAI-style API and host the
# same models, so only the address, the key and the limits change. Set LLM_PROVIDER in .env:
#   groq        Groq (GROQ_API_KEY). Free plan: 8K tokens a minute, 200K a day for gpt-oss-120b
#   openrouter  OpenRouter (OPENROUTER_API_KEY), pay-as-you-go from credits; every call is
#               pinned to OPENROUTER_PROVIDERS, in that order, and never sent anywhere else
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
PROVIDERS = {
    "groq": {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "tokens_per_day": 200_000},
    "openrouter": {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "tokens_per_day": None},  # no daily cap: spending is capped by credits and the key's limit
}
if LLM_PROVIDER not in PROVIDERS:
    raise ValueError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)} (got '{LLM_PROVIDER}')")
LLM_BASE_URL = PROVIDERS[LLM_PROVIDER]["base_url"]
LLM_API_KEY = os.getenv(PROVIDERS[LLM_PROVIDER]["key_env"], "")
if LLM_PROVIDER == "openrouter" and not LLM_API_KEY:
    LLM_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")  # also accepted, spelled with an underscore
LLM_KEY_ENV = PROVIDERS[LLM_PROVIDER]["key_env"]  # named in "key not set" messages

# OpenRouter hosts to use, first choice first (slugs from openrouter.ai/openai/gpt-oss-120b):
# Crusoe (full precision, ~250 tokens/s, 99.9% up), then Groq. Pinned so a whole evaluation
# runs on the same setup; no other host is ever used.
# Not CoreWeave: its gpt-oss puts the model's thinking inside the JSON answer ({"analysis": ...})
# instead of returning it separately, so Scout finds no meanings (tested 2026-11-03).
OPENROUTER_PROVIDERS = ["crusoe", "groq"]

# models: the same names on Groq and OpenRouter
LLM_MODEL_RAG = "openai/gpt-oss-20b"
LLM_MODEL_SQL = "openai/gpt-oss-120b"
LLM_MODEL_SUMMARY = "openai/gpt-oss-120b"  # offline table summaries (Text-to-SQL Version-3)

# Text-to-SQL Version-4: SQL is generated in rounds of 3 calls and compared; never more than this per question
SQL_CALLS_PER_ROUND = 3
MAX_LLM_CALLS_PER_QUESTION = 6
SQL_TEMPERATURE = 0.7  # some variety, so agreeing answers actually mean something
# Version-23: send the SQLite refresher with every SQL-writing call. Off, as it hasn't helped (on: SQL_REFRESHER=1 in .env)
SQL_REFRESHER = os.getenv("SQL_REFRESHER", "0").strip() != "0"

# Scout Version-8: performance limits for heavy queries
MAX_RESULT_ROWS = 200            # rows kept per query; fetch one more to know if there are more
PAGE_SIZE = 10                   # rows per page in the UI
QUERY_TIMEOUT_SECONDS = 5        # a query running longer than this is stopped
MAX_ESTIMATED_READS = 1_000_000  # rows a query may read, estimated from its plan before running
# Version-20: off for now. The estimate can be far off (BIRD #1482: ~1.9M estimated, ran in 0.03 s),
# so a query over the limit runs anyway, with a warning, and the time limit (QUERY_TIMEOUT_SECONDS)
# is the real guard. True brings back Version-8's refusal.
BLOCK_ESTIMATED_READS = False
RESULT_CACHE_SIZE = 10           # most recent results kept for paging (least recently used is dropped)
ANSWER_CACHE_SIZE = 10           # Version-12: recent answers reused when the same question comes back

# Version-14: evaluation runs started from the Evaluation tab. Rough cost of one question in the
# app's flow (meaning + steps + 3 SQL tries), measured on the practice database; used only for
# the estimate shown before a run starts.
EVAL_TOKENS_PER_QUESTION = 8_000
EVAL_CALLS_PER_QUESTION = 5.5
EVAL_SECONDS_PER_QUESTION = 20
# the provider's daily token limit (Groq free plan: 200K for gpt-oss-120b; OpenRouter: none); the estimate warns past it
PROVIDER_TOKENS_PER_DAY = PROVIDERS[LLM_PROVIDER]["tokens_per_day"]
# a rate-limited question waits this long and tries again, at most EVAL_RATE_LIMIT_TRIES times
EVAL_RATE_LIMIT_WAIT_SECONDS = 60
EVAL_RATE_LIMIT_TRIES = 3
QUERY_CONNECTIONS = 4            # queries that can run at the same time (one slow query doesn't block the rest)

# Version-18: BIRD Mini-Dev, unzipped from BIRD's minidev.zip and left exactly as downloaded.
# Kept outside data/text_to_sql/ on purpose: Scout attaches every .sqlite file in that folder,
# so BIRD's databases (and their questions) stay out of the app until an evaluation run uses them.
BIRD_MINI_DEV_DIR = ROOT_DIR / "data/benchmarks/bird/minidev/MINIDEV"
# Version-21: BIRD Mini-Dev with annotation errors fixed (Jin, Choi, Zhu, Kang, PVLDB 19: "Pervasive
# Annotation Errors Break Text-to-SQL Benchmarks and Leaderboards"; github.com/uiuc-kang-lab/text_to_sql_benchmarks)
BIRD_CORRECTED_DIR = ROOT_DIR / "data/benchmarks/bird/corrected"

# Version-20: the schema catalogs (one folder per database: catalog.json + values.sqlite), built
# offline from the data when a database is first opened, rebuilt only when it changes. Generated,
# so git-ignored; rebuilt any time with: python -m scripts.text_to_sql.onboard
CATALOG_DIR = ROOT_DIR / "data/catalogs"

# Text-to-SQL Version-5: the user confirms what the question means first; at most this many sets of options
MAX_CLARIFY_ROUNDS = 3
# when the checks find the confirmed meaning asks for a value the data doesn't have, the meaning
# is asked again with the real values: at most this many meanings in all (the first + 1 more)
MAX_MEANING_TRIES = 2
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
# are billed as output. OpenRouter reports each call's exact cost, which is used instead when
# present; these are the fallback and the basis of estimates.
#   Groq: on-demand prices as listed on 2026-11-03 (third-party trackers; check Groq's console)
#   OpenRouter: Crusoe's prices for gpt-oss-120b, its first host (Groq through OpenRouter: 0.15 / 0.60)
PRICES_BY_PROVIDER = {
    "groq": {"openai/gpt-oss-120b": (0.15, 0.60), "openai/gpt-oss-20b": (0.075, 0.30)},
    "openrouter": {
        "openai/gpt-oss-120b": (0.05, 0.25),
        "openai/gpt-oss-20b": (0.075, 0.30),  # Groq's price: Crusoe doesn't host it (OpenRouter, 2026-11-05)
        "meta-llama/llama-3.1-8b-instruct": (0.05, 0.08),
        "google/gemma-4-31b-it": (0.14, 0.40),
        "meta-llama/llama-3.3-70b-instruct": (0.59, 0.79),
        # the Mixed flow, on Crusoe (OpenRouter's list, 2026-11-06)
        "z-ai/glm-5.3-flash": (0.15, 0.50),
        "deepseek/deepseek-v4.1-flash": (0.29, 1.20),
        "nvidia/nemotron-3.5-super-vl": (0.10, 0.40),
    },
}
LLM_PRICES = PRICES_BY_PROVIDER[LLM_PROVIDER]

# Version-22: the Mixed flow's voters: three labs, each answering every question once. OpenRouter only.
# GLM 5.3 Flash was removed after its first run: it thought for 131,072 tokens and 460 seconds on one rewrite.
MIXED_MODELS = [
    {"id": "openai/gpt-oss-120b", "voter": "GPT-OSS 120B"},
    {"id": "deepseek/deepseek-v4.1-flash", "voter": "DeepSeek V4.1 Flash"},
    {"id": "nvidia/nemotron-3.5-super-vl", "voter": "Nemotron 3.5 Super"},
]

# Version-22: the models an evaluation can use, smallest to largest (size, not measured quality). Each is
# hosted by one of OPENROUTER_PROVIDERS, checked against OpenRouter's list on 2026-11-05, so no new
# host sees anything. Their ids only work through OpenRouter, except gpt-oss on Groq directly.
EVAL_MODELS = [
    {"id": "meta-llama/llama-3.1-8b-instruct", "name": "Llama 3.1 8B", "host": "Groq", "openrouter_only": True},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "host": "Groq", "openrouter_only": False},
    {"id": "google/gemma-4-31b-it", "name": "Gemma 4 31B", "host": "Crusoe", "openrouter_only": True},
    {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "host": "Groq", "openrouter_only": True},
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "host": "Crusoe", "openrouter_only": False},
]

# Privacy: Groq sees the question and the schema. When False, Scout never sends data rows:
# no sample rows when writing table descriptions, no result rows in the disagreement feedback.
# (Scribe always sends the passages it answers from: that's how RAG works.)
SHARE_ROWS_WITH_LLM = True

# Optional login: when set, the whole app asks for this password (any username) before
# anything loads. Empty = no login, fine while the app only runs on your own machine.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# Hugging Face: RAG evaluation scripts
HF_TOKEN = os.getenv("HF_TOKEN", "")
