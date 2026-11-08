"""Offline tests for the LLM proposal validator; no API calls are made."""
import json
from pathlib import Path

from llm_selector import build_lines, select_promos
from parser.txt_parser import parse_txt_content


def _episode():
    raw = (Path(__file__).resolve().parents[1] / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    block = raw.split("EPISODE NUMBER: Episode 1", 1)[1]
    block = "EPISODE NUMBER: Episode 1" + block.split("EPISODE NUMBER: Episode 2", 1)[0]
    return parse_txt_content(block, "synthetic_ep1.txt")


def test_llm_selection_uses_real_line_timestamps_and_valid_duration():
    ep = _episode()
    lines = build_lines(ep)
    assert len(lines) >= 2
    start = lines[0]
    end = next(line for line in lines if 20 <= line["end"] - start["start"] <= 60)
    response = {"promos": [{
        "start_line": start["id"], "end_line": end["id"],
        "hook_line": start["id"], "cliffhanger_line": end["id"],
        "hook": "Opening beat", "cliffhanger": "Unresolved beat", "why": "Test",
        "scores": {"hook": 7, "cliffhanger": 7, "standalone": 6, "emotion": 6, "curiosity": 6},
    }]}
    candidates, rejected = select_promos(ep, lambda _system, _user: json.dumps(response), max_promos=1)
    assert not rejected
    assert len(candidates) == 1
    assert 20 <= candidates[0].duration <= 60
    assert candidates[0].start == start["start"]
    assert candidates[0].end == end["end"]


def test_llm_unknown_line_id_is_rejected_not_used_as_timestamp():
    ep = _episode()
    response = {"promos": [{"start_line": "L999", "end_line": "L1000"}]}
    candidates, rejected = select_promos(ep, lambda _system, _user: json.dumps(response), max_promos=1)
    assert candidates == []
    assert rejected and "unknown line id" in rejected[-1]["reason"]


def test_llm_unknown_hook_or_cliffhanger_line_id_is_rejected():
    ep = _episode()
    lines = build_lines(ep)
    first = lines[0]
    end = next(line for line in lines if 20 <= line["end"] - first["start"] <= 60)
    response = {"promos": [{
        "start_line": first["id"], "end_line": end["id"],
        "hook_line": "L999", "cliffhanger_line": end["id"],
        "scores": {"hook": 8, "cliffhanger": 8, "standalone": 8, "emotion": 8, "curiosity": 8},
    }]}
    candidates, rejected = select_promos(ep, lambda _system, _user: json.dumps(response), max_promos=1)
    assert candidates == []
    assert rejected and "unknown hook line id" in rejected[-1]["reason"]
