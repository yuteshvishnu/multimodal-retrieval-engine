"""
One place to create the LLM client (Groq or OpenRouter, both OpenAI-compatible: LLM_PROVIDER in
config.py), so every call in the app has the same time limit, retries and provider settings.

    timeout       a call that gets no answer in LLM_TIMEOUT_SECONDS fails instead of hanging
    max_retries   rate limits (429), server errors (5xx), timeouts and dropped connections are
                  retried with exponential backoff, honouring the provider's "try again in Xs" hint
    OpenRouter    (Version-16) every call is pinned to OPENROUTER_PROVIDERS, in order, with no
                  fallback to other hosts, and asks for the call's exact cost in its usage
"""
import time
from typing import Dict, Optional

from openai import OpenAI

from backend.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRIES, LLM_PROVIDER, LLM_TIMEOUT_SECONDS, OPENROUTER_PROVIDERS,
)


def make_client(api_key: str = LLM_API_KEY) -> Optional[OpenAI]:
    """The shared client settings; None when no API key is set."""
    if not api_key:
        return None
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )


def chat(client: OpenAI, tag: Optional[Dict] = None, **kwargs):
    """
    client.chat.completions.create, recorded for the Dashboard: tokens (from the API's own
    count), time and price, tagged with the question and the job it did:
        tag = {"question_id": ..., "stage": "meaning" | "sql" | "answer" | "table_description", "round": n}
    """
    from backend.usage import usage_store  # here, so tests can swap the store

    if LLM_PROVIDER == "openrouter":
        kwargs["extra_body"] = {
            **kwargs.get("extra_body", {}),
            "provider": {"order": OPENROUTER_PROVIDERS, "allow_fallbacks": False},
            "usage": {"include": True},  # the call's exact cost, in credits ($)
        }
    start = time.monotonic()
    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception:
        usage_store.record_call(tag, kwargs.get("model", ""), None, time.monotonic() - start, "error")
        raise
    usage_store.record_call(tag, kwargs.get("model", ""), completion.usage, time.monotonic() - start, "ok")
    return completion
