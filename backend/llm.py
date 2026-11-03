"""
One place to create the LLM client (Groq, OpenAI-compatible API), so every call in the app
has the same time limit and retries.

    timeout       a call that gets no answer in LLM_TIMEOUT_SECONDS fails instead of hanging
    max_retries   rate limits (429), server errors (5xx), timeouts and dropped connections are
                  retried with exponential backoff, honouring Groq's "try again in Xs" hint
"""
from typing import Optional

from openai import OpenAI

from backend.config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS


def make_client(api_key: str = GROQ_API_KEY) -> Optional[OpenAI]:
    """The shared client settings; None when no API key is set."""
    if not api_key:
        return None
    return OpenAI(
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
