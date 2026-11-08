"""
llm_selector.py
===============
The LLM CHOOSES promo windows; plain code VALIDATES them.

Flow per episode
  1. Merge dialogue + camera/action lines into one numbered, timestamped list.
  2. Ask the model for up to N promos expressed as LINE IDs (never raw times).
  3. Code maps line IDs -> real timestamps, so every clip starts and ends on a
     real line. Code enforces 20-60 s, checks hook is at the start and the
     cliffhanger at the end, and records every rejection with a reason.

Output is the project's existing PromoCandidate, so scoring/ranking/export/UI
keep working unchanged.
"""
import json
import re
from typing import Callable, Dict, List, Optional, Tuple

import config
from analysis.screenplay_analyzer import PromoCandidate
from analysis.overlap import is_duplicate
from llm_client import LLMError, LLMFatalError
from parser.timestamp_parser import sec_to_ts
from parser.txt_parser import Episode

HOOK_WINDOW_S  = config.HOOK_WINDOW_SECONDS
CLIFF_WINDOW_S = config.CLIFFHANGER_WINDOW_SECONDS

SYSTEM_PROMPT = f"""You are a short-drama promo editor for TikTok / Instagram Reels / YouTube Shorts.
You get ONE episode screenplay. Every line has an ID and timestamps.
Pick up to {{n}} promo clips. A promo is one continuous run of lines, {config.MIN_PROMO_DURATION}-{config.MAX_PROMO_DURATION} seconds long.
Start the clip at the hook line and end it on the cliffhanger line. Do not include a long lead-in before the hook or extra resolved material after the cliffhanger.
 1. HOOK: opens with something that stops a scrolling viewer in the first {config.HOOK_WINDOW_SECONDS:.0f} seconds
    (a physical event, a meaningful reveal, a consequential accusation, or a compelling line). Not a greeting, camera direction, or scene-setting.
 2. CLIFFHANGER: ends at an UNRESOLVED moment (a threat, a question, a half-made reveal).
    Do not end after the conflict is resolved. A line merely ending in '?' or '!' is NOT enough.
 3. STANDALONE: someone who never saw the show understands who is fighting and what is at stake.
Use ONLY the line IDs given. Never invent lines or times. Different clips must not overlap much.
Return ONLY JSON, no prose, in this exact shape:
{{"promos":[{{"start_line":"L003","end_line":"L019","hook_line":"L003","cliffhanger_line":"L019",
"hook":"<why the first seconds grab>","cliffhanger":"<what is left unresolved>","why":"<one line>",
"scores":{{"hook":0-10,"cliffhanger":0-10,"standalone":0-10,"emotion":0-10,"curiosity":0-10}}}}]}}
Be strict with scores: 8+ only for clips you would really post. If the episode has no good promo, return fewer or an empty list."""


# --------------------------------------------------------------------------
# 1. Build the numbered line list
# --------------------------------------------------------------------------
def build_lines(ep: Episode) -> List[dict]:
    items = []
    for sc in ep.scenes:
        for d in sc.dialogue:
            tone = f" ({d.tone})" if d.tone else ""
            items.append(dict(start=d.start, end=d.end, scene=sc.number, kind="dialogue",
                              text=f'{d.character}{tone}: "{d.text}"', raw=d.text))
        for a in sc.actions:
            items.append(dict(start=a.start, end=a.end, scene=sc.number, kind="action",
                              text=f"[ACTION] {a.description}", raw=a.description))
    items.sort(key=lambda x: (x["scene"], x["start"], x["end"]))
    for i, it in enumerate(items, 1):
        it["id"] = f"L{i:03d}"
    return items


def render_script(lines: List[dict]) -> str:
    return "\n".join(f'{l["id"]} [{sec_to_ts(l["start"])}-{sec_to_ts(l["end"])}] {l["text"]}'
                     for l in lines)


# --------------------------------------------------------------------------
# 2. Parse the model's JSON defensively
# --------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _num(v, default=0.0) -> float:
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# 3. Validate one proposed promo -> PromoCandidate or a rejection reason
# --------------------------------------------------------------------------
def _validate(p: dict, ep: Episode, lines: List[dict], idx: Dict[str, int],
              n: int, snap: bool = False) -> Tuple[Optional[PromoCandidate], str]:
    try:
        a, b = idx[p["start_line"]], idx[p["end_line"]]
    except KeyError as e:
        return None, f"unknown line id {e}"          # model invented an ID
    if a > b:
        return None, "start_line after end_line"

    # Short clip? extend forward line by line (code, not model, fixes length).
    while (lines[b]["end"] - lines[a]["start"]) < config.MIN_PROMO_DURATION and b + 1 < len(lines):
        b += 1
    start, end = lines[a]["start"], lines[b]["end"]
    dur = end - start
    if dur < config.MIN_PROMO_DURATION:
        return None, f"too short ({dur:.1f}s) even after extending"
    if dur > config.MAX_PROMO_DURATION:
        return None, f"too long ({dur:.1f}s)"

    flags: List[str] = []
    hook_id = p.get("hook_line")
    cliff_id = p.get("cliffhanger_line")
    if not hook_id or not cliff_id:
        return None, "missing hook_line or cliffhanger_line ID"
    if hook_id and hook_id not in idx:
        return None, f"unknown hook line id {hook_id}"
    if cliff_id and cliff_id not in idx:
        return None, f"unknown cliffhanger line id {cliff_id}"
    h = idx[hook_id] if hook_id else a
    c = idx[cliff_id] if cliff_id else b
    if not (a <= h <= b) or not (a <= c <= b):
        return None, "hook or cliffhanger line is outside the proposed window"
    if h > c:
        return None, "hook line is after cliffhanger line"

    adjusted = False
    hook_late = lines[h]["start"] - start > HOOK_WINDOW_S
    cliff_late = end - lines[c]["end"] > CLIFF_WINDOW_S
    if hook_late or cliff_late:
        if not snap:
            if hook_late:
                return None, f"hook line is not within first {HOOK_WINDOW_S:.0f}s of clip"
            return None, f"cliffhanger line is not within last {CLIFF_WINDOW_S:.0f}s of clip"
        if hook_late:
            start = lines[h]["start"]
        if cliff_late:
            end = lines[c]["end"]
        dur = end - start
        if not config.MIN_PROMO_DURATION <= dur <= config.MAX_PROMO_DURATION:
            return None, "snap would break duration"
        adjusted = True

    final_indexes = [i for i, line in enumerate(lines)
                     if line["start"] >= start and line["end"] <= end]
    if not final_indexes:
        return None, "no parsed lines inside final clip"
    first_idx, last_idx = final_indexes[0], final_indexes[-1]
    s = p.get("scores", {}) or {}
    quotes = [f'EP{ep.number} [{sec_to_ts(l["start"])}] {l["text"][:100]}'
              for l in lines[first_idx:last_idx + 1] if l["kind"] == "dialogue"][:3]
    cand = PromoCandidate(
        promo_id=f"EP{ep.number:02d}-LLM-{n:02d}", episode=ep.number,
        scene_range=(lines[first_idx]["scene"], lines[last_idx]["scene"]),
        start=round(start, 2), end=round(end, 2), duration=round(dur, 2),
        start_ts=sec_to_ts(start), end_ts=sec_to_ts(end), adjusted=adjusted,
        hook_score=_num(s.get("hook")), hook_type="LLM",
        hook_evidence=f'{lines[h]["text"][:160]} — {p.get("hook", "")}'.strip(" —"),
        cliffhanger_score=_num(s.get("cliffhanger")), cliffhanger_type="LLM",
        cliffhanger_evidence=f'{lines[c]["text"][:160]} — {p.get("cliffhanger", "")}'.strip(" —"),
        standalone_score=_num(s.get("standalone")),
        standalone_explanation="LLM-rated standalone clarity (0-10).",
        emotional_intensity=_num(s.get("emotion")),
        viewer_curiosity=_num(s.get("curiosity")),
        why_it_works=str(p.get("why", "")), evidence=" | ".join(quotes),
    )
    cand.quality_flags = flags
    return cand, ""


# --------------------------------------------------------------------------
# 4. Public entry point
# --------------------------------------------------------------------------
def select_promos(ep: Episode, client: Callable[[str, str], str],
                  max_promos: int = 3, snap: bool = False) -> Tuple[List[PromoCandidate], List[dict]]:
    """Returns (valid candidates, rejection log). Nothing is silently dropped."""
    lines = build_lines(ep)
    rejected: List[dict] = []
    if not lines:
        return [], [{"episode": ep.number, "reason": "no parsed lines"}]
    idx = {l["id"]: i for i, l in enumerate(lines)}

    user = (f"EPISODE {ep.number}\nLOGLINE: {ep.logline}\nCONTINUITY: {ep.continuity}\n\n"
            f"SCRIPT:\n{render_script(lines)}")
    system = SYSTEM_PROMPT.replace("{n}", str(max_promos))

    data = None
    for attempt in range(2):                          # one retry on bad JSON
        try:
            data = _extract_json(client(system, user))
            break
        except LLMFatalError:
            raise                                     # bad key/model/URL: retrying cannot help
        except LLMError as e:
            # Transport problem already retried inside the client; do not wait through it twice.
            rejected.append({"episode": ep.number, "transport_error": True,
                             "reason": f"bad model output (try {attempt+1}): {e}"})
            break
        except Exception as e:
            rejected.append({"episode": ep.number, "reason": f"bad model output (try {attempt+1}): {e}"})
    if not data:
        return [], rejected

    out: List[PromoCandidate] = []
    for n, p in enumerate(data.get("promos", [])[:max_promos], 1):
        cand, why = _validate(p, ep, lines, idx, n, snap=snap)
        if cand is None:
            rejected.append({"episode": ep.number, "proposal": p, "reason": why})
        elif any(is_duplicate(cand, o) for o in out):
            rejected.append({"episode": ep.number, "proposal": p, "reason": "overlaps a better clip"})
        else:
            out.append(cand)
    return out, rejected
