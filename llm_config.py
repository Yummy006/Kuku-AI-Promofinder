"""Single place that resolves the LLM key, endpoint and model.

Order: environment variables (LLM_* first, then OPENAI_* as a fallback), then
Streamlit secrets (Streamlit Community Cloud). The CLI, the Streamlit app, and
llm_client all use this, so a credential is either available everywhere or nowhere.
The API key is never logged or printed.
"""
import os
from typing import NamedTuple

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class LLMSettings(NamedTuple):
    api_key: str
    base_url: str
    model: str


def _from_secrets(*names: str) -> str:
    try:
        import streamlit as st
        for name in names:
            value = st.secrets.get(name)
            if value:
                return str(value)
    except Exception:  # streamlit missing, or no secrets file configured
        pass
    return ""


def _setting(env_names, secret_names, default=""):
    for name in env_names:
        if os.getenv(name):
            return os.getenv(name)
    return _from_secrets(*secret_names) or default


def resolve_llm_settings() -> LLMSettings:
    return LLMSettings(
        api_key=_setting(("LLM_API_KEY", "OPENAI_API_KEY"), ("LLM_API_KEY", "OPENAI_API_KEY")),
        base_url=_setting(("LLM_BASE_URL", "OPENAI_BASE_URL"), ("LLM_BASE_URL", "OPENAI_BASE_URL"), DEFAULT_BASE_URL),
        model=_setting(("LLM_MODEL", "OPENAI_MODEL"), ("LLM_MODEL", "OPENAI_MODEL"), DEFAULT_MODEL),
    )
