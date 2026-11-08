"""
export/json_export.py
=====================
Export PromoCandidate results and quality reports to JSON.
Returns bytes for Streamlit download buttons.
"""

import json
import os
from typing import List, Dict, Any
from analysis.screenplay_analyzer import PromoCandidate
from export.segments import get_segments


def _candidate_to_dict(c: PromoCandidate) -> Dict[str, Any]:
    return {
        "rank":                    c.rank,
        "promo_id":                c.promo_id,
        "episode":                 c.episode,
        "scene_range":             list(c.scene_range),
        "start_ts":                c.start_ts,
        "end_ts":                  c.end_ts,
        "segments":                get_segments(c),
        "adjusted":                bool(getattr(c, "adjusted", False)),
        "duration_sec":            c.duration,
        "hook": {
            "score":    c.hook_score,
            "type":     c.hook_type,
            "evidence": c.hook_evidence,
        },
        "cliffhanger": {
            "score":    c.cliffhanger_score,
            "type":     c.cliffhanger_type,
            "evidence": c.cliffhanger_evidence,
        },
        "standalone": {
            "score":       c.standalone_score,
            "explanation": c.standalone_explanation,
        },
        "emotional_intensity": c.emotional_intensity,
        "viewer_curiosity":    c.viewer_curiosity,
        "total_score":         c.total_score,
        "recommendation":      c.recommendation,
        "why_it_works":        c.why_it_works,
        "evidence":            c.evidence,
        "quality_flags":       c.quality_flags,
        "human_review":        c.human_review,
    }


def candidates_to_json_bytes(candidates: List[PromoCandidate]) -> bytes:
    """Return JSON bytes of all candidates."""
    data = [_candidate_to_dict(c) for c in candidates]
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def quality_report_to_json_bytes(report: Dict[str, Any],
                                   candidates: List[PromoCandidate]) -> bytes:
    """Return JSON bytes of the full quality report."""
    data = {
        "summary": report,
        "candidates": [_candidate_to_dict(c) for c in candidates],
    }
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def save_outputs(candidates: List[PromoCandidate],
                 quality_report: Dict[str, Any],
                 output_dir: str = "outputs") -> None:
    """
    Save JSON and quality-report to the outputs/ folder.
    Used when running the analysis from CLI or after a Streamlit session.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ranked results
    results_path = os.path.join(output_dir, "promo_candidates.json")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(candidates_to_json_bytes(candidates).decode("utf-8"))

    # Quality report
    qr_path = os.path.join(output_dir, "quality_report.json")
    with open(qr_path, "w", encoding="utf-8") as f:
        f.write(quality_report_to_json_bytes(quality_report, candidates).decode("utf-8"))

    print(f"[INFO] Outputs saved → {results_path}")
    print(f"[INFO] Quality report → {qr_path}")
