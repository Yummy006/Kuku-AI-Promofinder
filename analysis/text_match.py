"""Shared word-boundary matching for screenplay keyword signals."""
import re
from typing import Iterable


def _inflected_forms(keyword: str):
    forms = {keyword, keyword + "s", keyword + "es", keyword + "ed", keyword + "ing"}
    if keyword.endswith("e"):
        forms.add(keyword + "d")
        forms.add(keyword[:-1] + "ing")
    if (len(keyword) >= 3 and keyword[-1] in "bddgmnprt"
            and keyword[-2] in "aeiou" and keyword[-3].isalpha()
            and keyword[-3] not in "aeiou"):
        forms.add(keyword + keyword[-1] + "ed")
        forms.add(keyword + keyword[-1] + "ing")
    return forms


def kw_hit(text: str, keywords: Iterable[str]) -> int:
    """Count inflected keyword phrases without matching inside other words."""
    tl = (text or "").lower()
    count = 0
    for keyword in keywords:
        kw = str(keyword).lower().strip()
        if not kw:
            continue
        forms = sorted(_inflected_forms(kw), key=len, reverse=True)
        pattern = r"(?<!\w)(?:" + "|".join(re.escape(form) for form in forms) + r")(?!\w)"
        if re.search(pattern, tl):
            count += 1
    return count
