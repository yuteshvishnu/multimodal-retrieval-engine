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
GROQ_MODEL_SUMMARY = "openai/gpt-oss-120b"  # offline table summaries (Text-to-SQL step 3)

# Text-to-SQL step 4: SQL is generated in rounds of 3 calls and compared; never more than this per question
SQL_CALLS_PER_ROUND = 3
MAX_LLM_CALLS_PER_QUESTION = 6
SQL_TEMPERATURE = 0.7  # some variety, so agreeing answers actually mean something

# Text-to-SQL step 5: the user confirms what the question means first; at most this many sets of options
MAX_CLARIFY_ROUNDS = 3

# Hugging Face: RAG evaluation scripts
HF_TOKEN = os.getenv("HF_TOKEN", "")
