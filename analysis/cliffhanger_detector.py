"""Detect unresolved ending beats; punctuation alone is never sufficient."""
import re
from typing import Tuple
import config
from analysis.text_match import kw_hit
from parser.txt_parser import Scene

DENIAL_AGREEMENT = re.compile(
    r"^\s*(?:i\s+)?(?:didn't|did not|don't|do not|wasn't|was not|"
    r"weren't|were not|can't|cannot|won't|will not|yes|no|okay|ok|"
    r"alright|all right|sure)\s*[.!?…]*\s*$", re.I
)
THREAT = re.compile(r"\b(?:kill|destroy|ruin|hunt|find you|i will find|i will come back|"
                    r"come for you|pay for this|you will regret|you'll regret|"
                    r"i will not stop|i will not rest|someone is coming|they're here|"
                    r"you are next|i'll get you|i will get you)\b", re.I)
REVEAL = re.compile(r"\b(?:the truth is|the real|your father|your mother|"
                    r"the killer|the evidence|i know who|it was you|"
                    r"i have proof|secret|identity|revealed)\b", re.I)
ACCUSATION = re.compile(r"\b(?:you lied|you betrayed|you killed|you stole|"
                        r"you did this|you are the one|i know what you did)\b", re.I)
CUT_MID_ACTION = re.compile(r"\b(?:reaches for|about to|raises the|aims at|"
                            r"door bursts|footsteps approach|someone enters|"
                            r"gunshot|suddenly|just as)\b", re.I)
UNANSWERED = re.compile(r"\b(?:who is|who are|what happened to|where is|"
                        r"why did|how did|what if|what will happen)\b", re.I)

def _ends_with_question_or_threat(text: str) -> bool:
    """Punctuation is a cue only when paired with unresolved semantic content."""
    t=text.strip()
    if not t or DENIAL_AGREEMENT.match(t):
        return False
    return bool(THREAT.search(t) or REVEAL.search(t) or ACCUSATION.search(t)
                or CUT_MID_ACTION.search(t) or (t.endswith("?") and UNANSWERED.search(t)))

def detect_cliffhanger(scene: Scene, window_start: float = None,
                       window_end: float = None) -> Tuple[float, str, str]:
    ws=window_start if window_start is not None else scene.start
    we=window_end if window_end is not None else scene.end
    cutoff=max(ws, we-config.CLIFFHANGER_WINDOW_SECONDS)
    late_dlg=[d for d in scene.dialogue if d.end >= cutoff and d.end <= we+0.5]
    late_act=[a for a in scene.actions if a.end >= cutoff and a.end <= we+0.5]

    # Explicit cut-to-black/music directions only count when they communicate
    # an actual interruption or unresolved transition.
    dir_score=0.0; dir_evidence=""
    for a in late_act:
        t=a.description.lower()
        if CUT_MID_ACTION.search(t):
            dir_score=5.5; dir_evidence=a.description[:160]; break

    dlg_score=0.0; dlg_evidence=""
    for d in reversed(late_dlg):
        text=d.text.strip()
        if DENIAL_AGREEMENT.match(text):
            continue
        if _ends_with_question_or_threat(text):
            signal_count=sum(bool(rx.search(text)) for rx in (THREAT,REVEAL,ACCUSATION,CUT_MID_ACTION,UNANSWERED))
            dlg_score=min(9.0, 4.5 + signal_count*1.5)
            dlg_evidence=f'{d.character}: "{d.text[:140]}"'
            break

    emo=0.0
    if dlg_score or dir_score:
        for d in late_dlg:
            for word, weight in config.EMOTION_KEYWORDS.items():
                if kw_hit(d.tone + " " + d.text, [word]):
                    emo = min(2.0, emo + weight * 0.2)
    total=min(10.0,max(dir_score,dlg_score)+emo)
    if dir_evidence:
        kind,evidence="Unresolved visual action",dir_evidence
    elif dlg_evidence:
        kind,evidence="Unresolved dialogue",dlg_evidence
    else:
        total=min(total,1.0)
        kind="Weak ending (no clear cliffhanger)"
        evidence=(f'{late_dlg[-1].character}: "{late_dlg[-1].text[:140]}"'
                  if late_dlg else (late_act[-1].description[:160] if late_act else
                                    "(No dialogue or action near clip end)"))
    return round(total,2),kind,evidence
