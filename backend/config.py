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

# Groq: RAG answers (Version-1) and Text-to-SQL (Version-2)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq models (see what your key can use: https://console.groq.com/docs/models)
GROQ_MODEL_RAG = "openai/gpt-oss-20b"
GROQ_MODEL_SQL = "openai/gpt-oss-120b"

# Hugging Face: RAG evaluation scripts
HF_TOKEN = os.getenv("HF_TOKEN", "")
