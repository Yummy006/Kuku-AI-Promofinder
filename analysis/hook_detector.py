"""Detect genuine opening hooks while excluding camera-only shot descriptions."""
import re
from typing import List, Tuple
import config
from analysis.text_match import kw_hit
from parser.txt_parser import Scene

CAMERA_PREFIX = re.compile(
    r"^\s*(?:(?:extreme\s+)?(?:medium|close[- ]?up|wide|long|two[- ]?shot|"
    r"overhead|tracking|dolly|handheld|over[- ]the[- ]shoulder|low[- ]angle|"
    r"high[- ]angle|aerial|establishing|point[- ]of[- ]view|pov)\s+shot\b|"
    r"(?:camera|the camera)\s+(?:pans|tilts|zooms|tracks|moves|pushes|pulls)\b)",
    re.I,
)
STRONG_EVENT = re.compile(
    r"\b(?:slaps?|strikes?|punch(?:es|ed)?|falls?|collapses?|reveals?|"
    r"screams?|throws?|shoves?|grabs?|draws?|shoots?|stabs?|kicks?|"
    r"crashes?|explodes?|lunges?|drags?|smashes?|attacks?|fires?)\b", re.I
)


def _is_camera_only(text: str) -> bool:
    """A shot/camera direction is not itself a hook without an event verb."""
    return bool(CAMERA_PREFIX.search(text)) and not bool(STRONG_EVENT.search(text))

def detect_hook(scene: Scene, window_start: float = None,
                window_end: float = None) -> Tuple[float, str, str]:
    ws = window_start if window_start is not None else scene.start
    we = window_end if window_end is not None else scene.end
    cutoff = min(we, ws + config.HOOK_WINDOW_SECONDS)
    early_dlg = [d for d in scene.dialogue if d.start <= cutoff and d.end >= ws]
    early_act = [a for a in scene.actions if a.start <= cutoff and a.end >= ws]

    action_score = 0.0; action_evidence = ""
    for act in early_act:
        if _is_camera_only(act.description):
            continue
        hits = kw_hit(act.description, config.HOOK_ACTION_KEYWORDS)
        if hits:
            action_score = min(10.0, action_score + hits * 2.5)
            action_evidence = act.description[:160]

    dlg_score = 0.0; dlg_evidence = ""
    for d in early_dlg:
        hits = kw_hit(d.text, config.HOOK_DIALOGUE_KEYWORDS)
        if hits:
            dlg_score = min(10.0, dlg_score + hits * 2.0)
            dlg_evidence = f'{d.character}: "{d.text[:140]}"'
        tone_hits = kw_hit(d.tone, list(config.EMOTION_KEYWORDS))
        dlg_score = min(10.0, dlg_score + tone_hits * 0.5)

    vis_score = 0.0; vis_evidence = ""
    for act in early_act:
        if _is_camera_only(act.description):
            continue
        hits = kw_hit(act.description, config.HOOK_VISUAL_KEYWORDS)
        if hits:
            vis_score = min(8.0, vis_score + hits * 1.5)
            vis_evidence = act.description[:160]

    # Later beats may be evidence of a promising scene, but cannot be labelled
    # as the opening hook. Keep this deliberately low.
    if action_score == 0 and dlg_score == 0 and vis_score == 0:
        for d in [d for d in scene.dialogue if d.start > cutoff and d.start >= ws and d.end <= we]:
            hits = kw_hit(d.text, config.HOOK_DIALOGUE_KEYWORDS)
            if hits:
                dlg_score = min(5.0, hits * 1.5)
                dlg_evidence = f'{d.character}: "{d.text[:140]}"'
                break
        for a in [a for a in scene.actions if a.start > cutoff and a.start >= ws and a.end <= we]:
            if _is_camera_only(a.description):
                continue
            hits = kw_hit(a.description, config.HOOK_ACTION_KEYWORDS)
            if hits:
                action_score = min(5.0, hits * 1.5)
                action_evidence = a.description[:160]
                break

    best = max(action_score, dlg_score, vis_score)
    if action_score == best and action_evidence:
        kind, evidence = "Action shock", action_evidence
    elif dlg_score == best and dlg_evidence:
        kind, evidence = "Shocking dialogue", dlg_evidence
    elif vis_evidence:
        kind, evidence = "Visual reveal", vis_evidence
    else:
        best = min(best, 1.5)
        kind = "Opening line (weak hook)"
        evidence = (f'{early_dlg[0].character}: "{early_dlg[0].text[:140]}"'
                    if early_dlg else (early_act[0].description[:160] if early_act else
                                       "(No dialogue or action in opening window)"))
    return round(best, 2), kind, evidence
