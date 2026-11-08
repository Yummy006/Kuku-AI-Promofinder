"""
screenplay_analyzer.py
======================
Orchestrates all scene-level analysis and promo-candidate generation.

For every episode it:
  1. Annotates each scene with hook / emotion / suspense / cliffhanger labels.
  2. Slides windows (multi-scene and sub-scene) across the episode to find
     every 20-60 s candidate.
  3. Returns a list of PromoCandidate dataclass objects, ready for scoring.

AI selection
------------
The optional LLM selector lives in llm_selector.py. This module remains the
reliable offline fallback and produces explainable rule-based candidates.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config
from parser.txt_parser import Episode, Scene
from parser.timestamp_parser import sec_to_ts, validate_duration
from analysis.hook_detector import detect_hook
from analysis.cliffhanger_detector import detect_cliffhanger
from analysis.context_analyzer import analyze_context
from analysis.overlap import is_duplicate


# ---------------------------------------------------------------------------
# Data class for a single promo candidate
# ---------------------------------------------------------------------------

@dataclass
class PromoCandidate:
    promo_id:      str
    episode:       int
    scene_range:   Tuple[int, int]    # (first_scene_no, last_scene_no)
    start:         float              # seconds
    end:           float              # seconds
    duration:      float              # seconds
    start_ts:      str                # MM:SS.ss display
    end_ts:        str

    # Analysis results
    hook_score:         float = 0.0
    hook_type:          str   = ""
    hook_evidence:      str   = ""

    cliffhanger_score:  float = 0.0
    cliffhanger_type:   str   = ""
    cliffhanger_evidence: str = ""

    standalone_score:   float = 0.0
    standalone_explanation: str = ""

    emotional_intensity: float = 0.0
    viewer_curiosity:    float = 0.0

    # Scoring (filled by promo_scorer)
    total_score:    float = 0.0
    rank:           int   = 0
    recommendation: str   = ""

    why_it_works:   str   = ""
    evidence:       str   = ""          # key lines from the screenplay

    # Quality flags (filled by quality_checker)
    quality_flags:  List[str] = field(default_factory=list)
    human_review:   str = ""            # "Good" / "Bad" / "Needs Improvement"
    segments: Optional[List[dict]] = None
    adjusted: bool = False


# ---------------------------------------------------------------------------
# Emotion & curiosity helpers
# ---------------------------------------------------------------------------

def _emotion_score(text: str) -> float:
    tl = text.lower()
    score = 0.0
    for word, weight in config.EMOTION_KEYWORDS.items():
        if word in tl:
            score += weight
    return min(10.0, score / 2.5)


def _curiosity_score(scenes: List[Scene], ws: float, we: float) -> float:
    """
    Viewer curiosity: driven by questions, mysteries, identity reveals,
    unresolved threats, betrayal signals.
    """
    curiosity_words = [
        "who are you", "what is this", "tell me the truth",
        "what are you hiding", "i don't understand",
        "why", "how could", "what happened",
        "secret", "mystery", "hidden", "reveal",
        "identity", "who is", "he is", "she is",
        "is he", "is she", "impossible", "i can't believe",
    ]
    combined = " ".join(s.all_text for s in scenes).lower()
    hits = sum(1 for kw in curiosity_words if kw in combined)
    return min(10.0, hits * 1.5)


# ---------------------------------------------------------------------------
# Window builders
# ---------------------------------------------------------------------------

def _multi_scene_windows(scenes: List[Scene]) -> List[Tuple[List[Scene], float, float]]:
    """All contiguous scene groups whose total duration fits 20-62 s."""
    results = []
    n = len(scenes)
    for i in range(n):
        for j in range(i, n):
            group = scenes[i:j+1]
            ws = group[0].start
            we = group[-1].end
            dur = we - ws
            if dur < config.MIN_PROMO_DURATION:
                continue
            if dur > config.MAX_PROMO_DURATION + 2:
                break
            results.append((group, ws, we))
    return results


def _sub_scene_windows(scene: Scene,
                        min_s: float,
                        max_s: float) -> List[Tuple[float, float]]:
    """
    Anchor-based sub-windows within a single scene.
    Anchors = scene start + up to 5 dialogue starts.
    Lengths tested: 25, 35, 45, 55 s.
    Hard cap: 30 windows per scene.
    """
    MAX_WIN = 30
    anchors = [scene.start]
    for d in scene.dialogue[:5]:
        if d.start > scene.start + 2:
            anchors.append(round(d.start, 1))

    # For long scenes add a stride
    if (scene.end - scene.start) > max_s * 1.1:
        t = scene.start
        while t + min_s <= scene.end:
            anchors.append(round(t, 1))
            t = round(t + 10.0, 1)

    windows = []
    for a in sorted(set(anchors)):
        for length in (25.0, 35.0, 45.0, 55.0):
            a_end = round(a + length, 1)
            actual_end = min(a_end, scene.end)
            if (actual_end - a) >= min_s:
                windows.append((a, actual_end))
        if len(windows) >= MAX_WIN:
            break

    return list(set(windows))[:MAX_WIN]


# ---------------------------------------------------------------------------
# LLM enhancement (optional)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rule-based "why it works" generator (always available)
# ---------------------------------------------------------------------------

def _rule_why_it_works(scenes: List[Scene], hook_type: str,
                        cliff_type: str, duration: float) -> str:
    combined = " ".join(s.all_text for s in scenes).lower()

    themes = []
    if re.search(r"\b(?:proposal|marry me|accept my hand)\b", combined):
        themes.append("an unexpected proposal")
    if re.search(r"\b(?:poison|poisoned|toxic substance)\b", combined):
        themes.append("a possible poisoning revelation")
    if re.search(r"\b(?:under arrest|constable|police|theft)\b", combined):
        themes.append("a legal accusation or arrest")
    if re.search(r"\b(?:fight|combat|strike|overpower|attack)\b", combined):
        themes.append("a physical confrontation")
    if re.search(r"\b(?:shove|cell|dungeon|chains|drag)\b", combined):
        themes.append("a confinement or escape threat")
    if re.search(r"\b(?:disowned|abandoned|betrayed|no longer trust)\b", combined):
        themes.append("a relationship rupture")
    if re.search(r"\b(?:secret|identity|truth|hidden)\b", combined):
        themes.append("a secret or identity reveal")
    theme_str = themes[0] if themes else "a high-stakes confrontation"
    return (f"Opens on {theme_str} ({hook_type}); "
            f"{duration:.0f}s of escalating tension that ends unresolved "
            f"({cliff_type}) — forcing any viewer to keep watching.")


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------
_id_counter = 0

def _new_id(ep_num: int) -> str:
    global _id_counter
    _id_counter += 1
    return f"EP{ep_num:02d}-PROMO-{_id_counter:03d}"


# ---------------------------------------------------------------------------
# Per-episode candidate finder
# ---------------------------------------------------------------------------

def _score_window(scenes: List[Scene], ws: float, we: float,
                   episode: Episode, is_sub: bool = False) -> Optional[PromoCandidate]:
    """Score a single time window. Returns None if below minimum quality."""
    dur = we - ws
    valid, reason = validate_duration(ws, we,
                                       config.MIN_PROMO_DURATION,
                                       config.MAX_PROMO_DURATION)
    if not valid:
        return None

    # Build pseudo-scenes filtered to the window when doing sub-scene analysis
    if is_sub:
        src = scenes[0]
        from parser.txt_parser import Scene as S, DialogueLine as DL, ActionLine as AL
        sub_dlg = [d for d in src.dialogue if d.start >= ws - 1.5 and d.end <= we + 1.5]
        sub_act = [a for a in src.actions  if a.start >= ws - 1.5 and a.end <= we + 1.5]
        raw = " ".join(d.text for d in sub_dlg) + " " + " ".join(a.description for a in sub_act)

        pseudo = S(number=src.number, heading=src.heading,
                    start=ws, end=we)
        pseudo.dialogue = sub_dlg
        pseudo.actions  = sub_act
        pseudo.raw_text = raw
        eval_scenes = [pseudo]
        scene_range = (src.number, src.number)
    else:
        eval_scenes = scenes
        scene_range = (scenes[0].number, scenes[-1].number)

    hook_score, hook_type, hook_evidence   = detect_hook(eval_scenes[0], ws, we)
    cliff_score, cliff_type, cliff_evidence = detect_cliffhanger(eval_scenes[-1], ws, we)
    ctx_score, ctx_explanation              = analyze_context(eval_scenes, ws, we)

    combined_text = " ".join(s.all_text for s in eval_scenes)
    emo   = _emotion_score(combined_text)
    curio = _curiosity_score(eval_scenes, ws, we)

    # Reject very low-quality candidates early
    if (hook_score + cliff_score) < 2.0:
        return None

    why = _rule_why_it_works(eval_scenes, hook_type, cliff_type, dur)

    # Evidence: top 2 dialogue lines
    evidence_lines = []
    for s in eval_scenes:
        for d in s.dialogue[:3]:
            evidence_lines.append(f'EP{episode.number} [{sec_to_ts(d.start)}] '
                                   f'{d.character}: "{d.text[:100]}"')
    evidence = " | ".join(evidence_lines[:3])

    return PromoCandidate(
        promo_id      = _new_id(episode.number),
        episode       = episode.number,
        scene_range   = scene_range,
        start         = round(ws, 2),
        end           = round(we, 2),
        duration      = round(dur, 2),
        start_ts      = sec_to_ts(ws),
        end_ts        = sec_to_ts(we),
        hook_score    = hook_score,
        hook_type     = hook_type,
        hook_evidence = hook_evidence,
        cliffhanger_score    = cliff_score,
        cliffhanger_type     = cliff_type,
        cliffhanger_evidence = cliff_evidence,
        standalone_score     = ctx_score,
        standalone_explanation = ctx_explanation,
        emotional_intensity  = round(emo, 2),
        viewer_curiosity     = round(curio, 2),
        why_it_works = why,
        evidence     = evidence,
    )


def _deduplicate(candidates: List[PromoCandidate]) -> List[PromoCandidate]:
    """Keep only the highest-scoring candidate when two overlap too much.

    Uses the same overlap policy as scoring.ranking (see analysis/overlap.py).
    """
    candidates.sort(key=lambda x: -(x.hook_score + x.cliffhanger_score))
    kept = []
    for cand in candidates:
        if not any(is_duplicate(k, cand) for k in kept):
            kept.append(cand)
    return kept


def analyze_episode(episode: Episode) -> List[PromoCandidate]:
    """
    Find all promo candidates within one episode.
    Returns a deduplicated list sorted by (hook+cliff) score desc.
    """
    global _id_counter
    _id_counter = 0  # reset per episode for clean IDs

    scenes = episode.scenes
    if not scenes:
        return []

    candidates = []

    # 1. Multi-scene windows
    for group, ws, we in _multi_scene_windows(scenes):
        p = _score_window(group, ws, we, episode, is_sub=False)
        if p:
            candidates.append(p)

    # 2. Sub-scene windows (only when ≤2 scenes or no multi-scene hits)
    if len(scenes) <= 2 or len(candidates) == 0:
        for scene in scenes:
            for ws, we in _sub_scene_windows(
                    scene, config.MIN_PROMO_DURATION, config.MAX_PROMO_DURATION):
                p = _score_window([scene], ws, we, episode, is_sub=True)
                if p:
                    candidates.append(p)

    return _deduplicate(candidates)


def analyze_all_episodes(episodes: List[Episode]) -> List[PromoCandidate]:
    """Analyze every episode and return a flat list of all candidates."""
    all_candidates = []
    for ep in episodes:
        all_candidates.extend(analyze_episode(ep))
    return all_candidates
