"""
config.py
=========
Central configuration for PromoFinder AI.
All weights, thresholds and feature flags live here so they can be
changed in one place without touching analysis or scoring code.
"""

import os

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME        = "PromoFinder AI"
APP_VERSION     = "1.2.0"
APP_DESCRIPTION = "AI-powered Microdrama Promo Discovery System"

# ---------------------------------------------------------------------------
# Promo duration limits (seconds)
# ---------------------------------------------------------------------------
MIN_PROMO_DURATION = 20   # seconds
HOOK_WINDOW_SECONDS = 3.0
CLIFFHANGER_WINDOW_SECONDS = 6.0
MAX_PROMO_DURATION = 60   # seconds

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 100)
# ---------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "hook_strength":      25,
    "cliffhanger_strength": 25,
    "standalone_context": 20,
    "emotional_intensity": 15,
    "viewer_curiosity":   15,
}

# ---------------------------------------------------------------------------
# Rank-based recommendation tiers (global, after deduplication).
POST_FIRST_COUNT = int(os.getenv("PROMO_POST_FIRST_COUNT", "10"))
POST_LATER_COUNT = int(os.getenv("PROMO_POST_LATER_COUNT", "20"))
TOP_LIST_EPISODE_CAP = int(os.getenv("PROMO_TOP_EPISODE_CAP", "2"))
DEDUP_OVERLAP_RATIO = 0.40

# ---------------------------------------------------------------------------
# Hook keyword lists (case-insensitive)
# ---------------------------------------------------------------------------
HOOK_ACTION_KEYWORDS = [
    "slap", "punch", "strike", "shove", "grab", "seize", "crash",
    "explode", "fire", "shoot", "lunges", "collapse",
    "throw", "drag", "reveal", "scream", "attack", "stab", "kick",
]
HOOK_DIALOGUE_KEYWORDS = [
    "you lied", "you betrayed", "you knew", "tell me the truth",
    "how could you", "what have you done", "you're under arrest",
    "you will pay", "it was you", "who are you", "what happened",
    "i have proof", "the truth is", "i know what you did",
]
HOOK_VISUAL_KEYWORDS = [
    "kneels", "drops to", "pulls out", "tears streaming", "blood",
    "bleeding", "wound", "fire", "flames", "shackled", "handcuffs",
]
CLIFFHANGER_DIALOGUE = [
    "not yet", "not finished", "i'm not done", "who is", "who are",
    "how did", "why did", "what happened", "what if", "what will happen",
    "this isn't over", "you'll regret", "i will find", "i will come back",
    "someone is coming", "they're here", "i have proof", "the truth is",
]
CLIFFHANGER_VISUAL = [
    "abrupt cut", "cut mid-action", "reaches for", "raises the", "aims at",
    "door bursts", "footsteps approach", "someone enters", "gunshot",
]


# ---------------------------------------------------------------------------
# Emotional intensity keywords
# ---------------------------------------------------------------------------
EMOTION_KEYWORDS = {
    "furious":    2.0, "enraged":     2.0, "thunderous": 2.0,
    "screaming":  2.0, "desperate":   1.5, "weeping":    1.0,
    "sobbing":    1.0, "chilling":    1.5, "sinister":   1.5,
    "deadly":     2.0, "lethal":      2.0, "sacrificial": 1.5,
    "defiant":    1.0, "resolute":    1.0, "shocked":    1.0,
    "stunned":    1.0, "gasping":     1.5, "terrified":  1.5,
    "panic":      1.5, "triumphant":  1.0, "devastated": 2.0,
    "heartbroken": 1.5, "betrayed":  1.5, "anguished":  1.5,
}

# ---------------------------------------------------------------------------
# Standalone context keywords (help cold viewers understand the scene)
# ---------------------------------------------------------------------------
STANDALONE_KEYWORDS = [
    "my husband", "my wife", "my family", "my mother", "my father",
    "my son", "my daughter", "you abandoned", "you left me",
    "who are you", "i don't know you", "under arrest", "wedding",
    "altar", "dungeon", "cell", "chains", "shackles",
]

# ---------------------------------------------------------------------------
# Romance / intimacy keywords (boost for Instagram/YT female audience)
# ---------------------------------------------------------------------------
ROMANCE_KEYWORDS = [
    "tender", "tenderly", "leans closer", "gazes", "breathless",
    "your touch", "devotion", "devoted",
]

# Optional show-specific additions are loaded only when explicitly opted in.
# Keep the default config show-agnostic; never commit confidential show terms.
def _load_optional_show_keywords():
    import json
    path = os.getenv("PROMO_SHOW_KEYWORDS")
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

_SHOW_KEYWORDS = _load_optional_show_keywords()
for _key in ("HOOK_ACTION_KEYWORDS", "HOOK_DIALOGUE_KEYWORDS", "HOOK_VISUAL_KEYWORDS",
             "CLIFFHANGER_DIALOGUE", "CLIFFHANGER_VISUAL", "STANDALONE_KEYWORDS", "ROMANCE_KEYWORDS"):
    if _key in _SHOW_KEYWORDS and isinstance(_SHOW_KEYWORDS[_key], list):
        globals()[_key] = list(globals().get(_key, [])) + [str(x) for x in _SHOW_KEYWORDS[_key]]


# ---------------------------------------------------------------------------
# LLM / AI provider (optional — rule-based fallback used when not set)
# ---------------------------------------------------------------------------
# Set this env var to enable AI-enhanced analysis:
#   export OPENAI_API_KEY="sk-..."
# Any OpenAI-compatible endpoint is supported.
from llm_config import resolve_llm_settings as _resolve_llm_settings
_LLM = _resolve_llm_settings()
OPENAI_API_KEY  = _LLM.api_key
OPENAI_MODEL    = _LLM.model
OPENAI_BASE_URL = _LLM.base_url

# If True the app will offer optional LLM-assisted selection.
USE_LLM = bool(OPENAI_API_KEY)
