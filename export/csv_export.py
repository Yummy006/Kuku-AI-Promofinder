"""
export/csv_export.py
====================
Export PromoCandidate results to CSV format.
Returns bytes so Streamlit's st.download_button can use it directly.
"""

import io
import csv
from typing import List
from analysis.screenplay_analyzer import PromoCandidate
from export.segments import format_segments


def candidates_to_csv_bytes(candidates: List[PromoCandidate]) -> bytes:
    """
    Convert a list of PromoCandidate objects to a UTF-8 CSV in memory.
    Returns raw bytes suitable for st.download_button.
    """
    fieldnames = [
        "rank", "promo_id", "episode", "scene_range",
        "start_ts", "end_ts", "duration", "segments", "adjusted",
        "hook_score", "hook_type", "hook_evidence",
        "cliffhanger_score", "cliffhanger_type", "cliffhanger_evidence",
        "standalone_score", "standalone_explanation",
        "emotional_intensity", "viewer_curiosity",
        "total_score", "recommendation",
        "why_it_works", "evidence", "quality_flags",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for c in candidates:
        writer.writerow({
            "rank":                    c.rank,
            "promo_id":                c.promo_id,
            "episode":                 c.episode,
            "scene_range":             f"{c.scene_range[0]}-{c.scene_range[1]}",
            "start_ts":                c.start_ts,
            "end_ts":                  c.end_ts,
            "duration":                c.duration,
            "segments":                format_segments(c),
            "adjusted":                 bool(getattr(c, "adjusted", False)),
            "hook_score":              c.hook_score,
            "hook_type":               c.hook_type,
            "hook_evidence":           c.hook_evidence,
            "cliffhanger_score":       c.cliffhanger_score,
            "cliffhanger_type":        c.cliffhanger_type,
            "cliffhanger_evidence":    c.cliffhanger_evidence,
            "standalone_score":        c.standalone_score,
            "standalone_explanation":  c.standalone_explanation,
            "emotional_intensity":     c.emotional_intensity,
            "viewer_curiosity":        c.viewer_curiosity,
            "total_score":             c.total_score,
            "recommendation":          c.recommendation,
            "why_it_works":            c.why_it_works,
            "evidence":                c.evidence,
            "quality_flags":           " | ".join(c.quality_flags),
        })

    return buf.getvalue().encode("utf-8")


def quality_report_to_csv_bytes(report: dict,
                                  candidates: List[PromoCandidate]) -> bytes:
    """
    Export quality-check flags per candidate to CSV.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["promo_id", "episode", "total_score",
                     "recommendation", "quality_flag", "human_review"])
    for c in candidates:
        if c.quality_flags:
            for flag in c.quality_flags:
                writer.writerow([c.promo_id, c.episode, c.total_score,
                                  c.recommendation, flag, c.human_review])
        else:
            writer.writerow([c.promo_id, c.episode, c.total_score,
                              c.recommendation, "PASS", c.human_review])
    return buf.getvalue().encode("utf-8")
