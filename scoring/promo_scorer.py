"""
scoring/promo_scorer.py
=======================
Converts raw analysis scores into a 0-100 composite score for each
PromoCandidate, then assigns a recommendation label.

All weights are read from config.SCORING_WEIGHTS so they can be
changed without touching this file.

Score formula (default weights sum to 100):
  hook_strength        × 25 %
  cliffhanger_strength × 25 %
  standalone_context   × 20 %
  emotional_intensity  × 15 %
  viewer_curiosity     × 15 %

Each sub-score comes in as 0-10; it is multiplied by (weight / 10)
to produce a contribution toward the 100-point total.
"""

from typing import List, Dict

import config
from analysis.screenplay_analyzer import PromoCandidate


def score_candidate(candidate: PromoCandidate,
                    weights: Dict[str, int] = None) -> PromoCandidate:
    """
    Compute the total_score and recommendation for a single candidate.

    Parameters
    ----------
    candidate : PromoCandidate (mutated in-place)
    weights   : Optional override dict with same keys as config.SCORING_WEIGHTS

    Returns
    -------
    The same PromoCandidate object with total_score and recommendation filled in.
    """
    w = weights if weights is not None else config.SCORING_WEIGHTS

    required = set(config.SCORING_WEIGHTS)
    if set(w) != required:
        raise ValueError(f"Weights must contain exactly: {sorted(required)}")
    if any(not isinstance(v, (int, float)) or v < 0 for v in w.values()):
        raise ValueError("Scoring weights must be non-negative numbers")

    # Validate weights sum to 100
    if sum(w.values()) != 100:
        raise ValueError(
            f"Scoring weights must sum to 100, got {sum(w.values())}"
        )

    # Each raw score is 0-10; contribution = raw_score * (weight / 10)
    # Clamp every component to the documented 0–10 range. This prevents an
    # external/model response from creating impossible composite scores.
    component = lambda value: max(0.0, min(10.0, float(value or 0.0)))
    score = (
        component(candidate.hook_score)         * (w["hook_strength"]        / 10) +
        component(candidate.cliffhanger_score)  * (w["cliffhanger_strength"] / 10) +
        component(candidate.standalone_score)   * (w["standalone_context"]   / 10) +
        component(candidate.emotional_intensity)* (w["emotional_intensity"]  / 10) +
        component(candidate.viewer_curiosity)   * (w["viewer_curiosity"]     / 10)
    )

    candidate.total_score = round(min(score, 100.0), 1)

    # No absolute score tiers: final recommendations are assigned only after
    # global ranking. A severe context/hook/cliff weakness remains a review gate.
    if candidate.standalone_score < 2 or candidate.hook_score < 3 or candidate.cliffhanger_score < 2:
        candidate.recommendation = "👁️ Needs Human Review (pre-ranking quality gate)"
    else:
        candidate.recommendation = "Awaiting global rank"

    return candidate


def score_all(candidates: List[PromoCandidate],
              weights: Dict[str, int] = None) -> List[PromoCandidate]:
    """Score every candidate in the list. Returns the same list (mutated)."""
    for c in candidates:
        score_candidate(c, weights)
    return candidates
