"""One overlap policy shared by analysis, LLM selection, ranking and quality checks.

Two clips from the same episode are duplicates when their overlap, measured as a
share of the SHORTER clip, is greater than config.DEDUP_OVERLAP_RATIO.
"""
import config


def overlap_ratio(a, b) -> float:
    """Overlap seconds divided by the shorter clip's duration (0.0 if undefined)."""
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shorter = min(max(a.duration, 0.0), max(b.duration, 0.0))
    return overlap / shorter if shorter else 0.0


def is_duplicate(a, b, limit=None) -> bool:
    """True when two clips from the same episode overlap more than the limit."""
    if a.episode != b.episode:
        return False
    limit = config.DEDUP_OVERLAP_RATIO if limit is None else limit
    return overlap_ratio(a, b) > limit
