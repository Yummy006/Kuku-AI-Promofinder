"""Shared segment representation for promo exports."""
from typing import Any, Dict, List


def get_segments(candidate: Any) -> List[Dict[str, Any]]:
    """Return explicit ordered segments, or the legacy single clip segment."""
    segments = getattr(candidate, "segments", None)
    if segments:
        return [dict(segment) for segment in segments]
    return [{
        "episode": candidate.episode,
        "start_ts": candidate.start_ts,
        "end_ts": candidate.end_ts,
    }]


def format_segments(candidate: Any) -> str:
    return " | ".join(
        f"EP{segment['episode']} {segment['start_ts']}–{segment['end_ts']}"
        for segment in get_segments(candidate)
    )
