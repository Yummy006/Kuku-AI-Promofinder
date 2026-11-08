"""
llm_client.py
=============
One tiny, swappable LLM client. No SDK needed (plain HTTPS, standard library).

Works with ANY OpenAI-compatible chat endpoint (OpenAI, Gemini's OpenAI-compat
endpoint, Groq, OpenRouter, Together, local Ollama/vLLM ...). To swap models
change only these env vars:

    LLM_BASE_URL   e.g. https://api.openai.com/v1
    LLM_MODEL      e.g. gpt-4o-mini
    LLM_API_KEY    your key

(OPENAI_* names are accepted as a fallback so your old setup still works.)
Credentials are resolved by llm_config.py (environment first, then Streamlit
secrets) and are never logged or included in error messages.

Failure behaviour
-----------------
* Retryable (network error, timeout, HTTP 408/429/5xx, malformed reply): retried
  with backoff, then raises LLMError.
* Not retryable (HTTP 400/401/403/404/422 ...: bad key, wrong model, wrong URL):
  raises LLMFatalError immediately, because retrying cannot fix it.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Callable

from llm_config import resolve_llm_settings

_settings = resolve_llm_settings()
BASE_URL = _settings.base_url
MODEL    = _settings.model
API_KEY  = _settings.api_key

# A "client" is just a function: (system_prompt, user_prompt) -> text.
LLMClient = Callable[[str, str], str]
LAST_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
USAGE_TOTAL = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# HTTP statuses worth retrying; every other 4xx is a configuration error.
RETRYABLE_STATUS = {408, 409, 425, 429}


class LLMError(RuntimeError):
    """The model call failed after retries (transient problem)."""


class LLMFatalError(LLMError):
    """The model call can never succeed as configured (bad key, model or URL)."""


def has_credentials() -> bool:
    """True when an API key is configured (env var or Streamlit secret)."""
    return bool(API_KEY)


def reset_usage():
    for key in USAGE_TOTAL:
        USAGE_TOTAL[key] = 0


def chat(system: str, user: str, *, retries: int = 3, timeout: int = 90) -> str:
    """Call the configured model. Temperature 0 for repeatable runs."""
    if not API_KEY:
        raise LLMFatalError("No LLM_API_KEY set. See llm_client.py header.")
    LAST_USAGE.update({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    body = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            usage = data.get("usage") or {}
            LAST_USAGE.update({
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            })
            for key in USAGE_TOTAL:
                USAGE_TOTAL[key] += LAST_USAGE[key]
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            # Only the status line is reported: provider bodies can echo key fragments.
            if e.code >= 400 and e.code < 500 and e.code not in RETRYABLE_STATUS:
                raise LLMFatalError(
                    f"HTTP {e.code} {e.reason} from {BASE_URL} (model '{MODEL}'): "
                    "check LLM_API_KEY, LLM_MODEL and LLM_BASE_URL") from None
            last_err = f"HTTP {e.code} {e.reason}"
        except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as e:
            # OSError covers URLError, timeouts and dropped connections.
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(2 ** attempt)          # simple backoff (helps free-tier rate limits)
    raise LLMError(f"LLM call failed after {retries} tries: {last_err}")
