"""
timestamp_parser.py
===================
Utilities for parsing and converting timestamps found in screenplay files.

Supported formats:
  MM:SS.ss   e.g. 01:23.50
  HH:MM:SS   e.g. 00:01:23
  SS.ss      e.g. 83.5
  plain int  e.g. 83

All functions return float seconds or raise ValueError with a clear message.
"""

import re
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level converters
# ---------------------------------------------------------------------------

def ts_to_sec(ts: str) -> float:
    """
    Convert a timestamp string to float seconds.

    Examples
    --------
    >>> ts_to_sec("01:23.50")
    83.5
    >>> ts_to_sec("00:01:23")
    83.0
    >>> ts_to_sec("83.5")
    83.5
    """
    ts = ts.strip()

    # HH:MM:SS  or  HH:MM:SS.ss
    hms = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", ts)
    if hms:
        h, m, s = float(hms.group(1)), float(hms.group(2)), float(hms.group(3))
        return h * 3600 + m * 60 + s

    # MM:SS  or  MM:SS.ss
    ms = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", ts)
    if ms:
        m, s = float(ms.group(1)), float(ms.group(2))
        return m * 60 + s

    # plain seconds (float or int)
    plain = re.match(r"^(\d+(?:\.\d+)?)$", ts)
    if plain:
        return float(plain.group(1))

    raise ValueError(f"Cannot parse timestamp: '{ts}'")


def sec_to_ts(sec: float) -> str:
    """
    Convert float seconds back to MM:SS.ss display format.

    Example
    -------
    >>> sec_to_ts(83.5)
    '01:23.50'
    """
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def duration_str(start_sec: float, end_sec: float) -> str:
    """Return a human-readable duration string like '35.0 s'."""
    d = end_sec - start_sec
    return f"{d:.1f} s"


# ---------------------------------------------------------------------------
# Extraction from raw text
# ---------------------------------------------------------------------------

# Matches  Start: (01:23.50) | End: (02:01.00)
_PAIR_RE = re.compile(
    r"Start:\s*\(([0-9:.]+)\)\s*\|?\s*End:\s*\(([0-9:.]+)\)",
    re.IGNORECASE,
)

# Matches any standalone timestamp like (01:23.50) or [01:23]
_STANDALONE_RE = re.compile(r"\(([0-9]{1,2}:[0-9]{2}(?:\.[0-9]+)?)\)")


def extract_timestamp_pair(line: str) -> Optional[Tuple[float, float]]:
    """
    Extract a (start_sec, end_sec) pair from a single text line.
    Returns None if no pair is found.
    """
    m = _PAIR_RE.search(line)
    if m:
        try:
            return ts_to_sec(m.group(1)), ts_to_sec(m.group(2))
        except ValueError:
            return None
    return None


def extract_all_timestamps(text: str):
    """
    Extract all standalone timestamps from a block of text.
    Returns a list of float seconds.
    """
    results = []
    for m in _STANDALONE_RE.finditer(text):
        try:
            results.append(ts_to_sec(m.group(1)))
        except ValueError:
            pass
    return results


def validate_duration(start: float, end: float,
                      min_sec: float = 20.0,
                      max_sec: float = 60.0) -> Tuple[bool, str]:
    """
    Check whether (end - start) falls within [min_sec, max_sec].
    Returns (is_valid, reason_string).
    """
    if end <= start:
        return False, f"End ({sec_to_ts(end)}) is not after Start ({sec_to_ts(start)})"
    dur = end - start
    if dur < min_sec:
        return False, f"Duration {dur:.1f}s is shorter than minimum {min_sec}s"
    if dur > max_sec:
        return False, f"Duration {dur:.1f}s is longer than maximum {max_sec}s"
    return True, "OK"
