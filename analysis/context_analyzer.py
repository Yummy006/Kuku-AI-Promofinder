"""
context_analyzer.py
===================
Evaluates how well a new viewer (who has never seen the show) can
understand a scene or clip without any prior episodes.

Returns a score 0-10 and a short explanation.
"""

from typing import List, Tuple

import config
from analysis.text_match import kw_hit
from parser.txt_parser import Scene




# ---------------------------------------------------------------------------
# Visual-clarity bonus
# Context carried by the *setting* alone (readable from camera descriptions)
# ---------------------------------------------------------------------------
VISUAL_CONTEXT_WORDS = [
    "wedding gown", "bridal", "cathedral", "altar",
    "dungeon", "cell", "chains", "shackles",
    "bedroom", "bedchamber", "ballroom", "courtroom",
    "bleeding", "blood", "wound", "fire", "flames",
    "crying", "tears",
    "guards", "constable", "arrest",
    "throne", "crown", "estate", "manor",
]


def analyze_context(scenes: List[Scene],
                    window_start: float = None,
                    window_end: float = None) -> Tuple[float, str]:
    """
    Score standalone context (0-10) for a list of scenes (or a sub-window).

    Returns
    -------
    (score_0_to_10, explanation_string)
    """
    ws = window_start if window_start is not None else (scenes[0].start if scenes else 0)
    we = window_end   if window_end   is not None else (scenes[-1].end  if scenes else 0)

    # Collect all text in the window
    # Restrict evidence to the requested clip window. A multi-scene candidate must
    # not receive context credit for dialogue outside its actual timestamps.
    parts = []
    for scene in scenes:
        parts.extend(d.text for d in scene.dialogue if d.end >= ws and d.start <= we)
        parts.extend(a.description for a in scene.actions if a.end >= ws and a.start <= we)
    combined = " ".join(parts).lower()

    # ── 1. Relational / expository dialogue keywords ──────────────────────
    kw_score = kw_hit(combined, config.STANDALONE_KEYWORDS) * 1.5

    # ── 2. Visual context from setting and actions ────────────────────────
    vis_score = kw_hit(combined, VISUAL_CONTEXT_WORDS) * 0.8

    # ── 3. Self-explanatory one-liners (very high standalone power) ───────
    self_explaining = [
        "you are under arrest", "under arrest", "you abandoned me",
        "you left me", "i have proof", "i know what you did",
        "this is my family", "this is my child", "you are my parent",
    ]
    se_score = kw_hit(combined, self_explaining) * 2.5

    total = min(10.0, kw_score + vis_score + se_score)

    # ── 4. Build explanation ──────────────────────────────────────────────
    if total >= 7:
        explanation = ("Strong standalone clarity — key relationships and "
                       "stakes are clearly stated in the dialogue.")
    elif total >= 4:
        explanation = ("Moderate context — setting and some dialogue help "
                       "orient a new viewer, but a title card could help.")
    elif total >= 2:
        explanation = ("Weak context — the scene is emotionally intense but "
                       "a cold viewer may not understand who the characters "
                       "are. Recommend adding a text overlay.")
    else:
        explanation = ("No standalone context — this clip requires prior "
                       "episode knowledge. Not suitable as a first promo.")

    return round(total, 2), explanation
