import json
import subprocess
import sys
from pathlib import Path

import pytest
from parser.txt_parser import Scene, ActionLine, DialogueLine
from analysis.hook_detector import detect_hook
from analysis.cliffhanger_detector import detect_cliffhanger
from analysis.screenplay_analyzer import PromoCandidate
from scoring.ranking import rank_candidates, top_n, overlap_ratio
from evaluation.compare import evaluate

def candidate(ep,start,end,score):
    return PromoCandidate(
        promo_id=f"EP{ep}-{start}", episode=ep, scene_range=(1,1),
        start=start,end=end,duration=end-start,start_ts=f"00:{start:05.2f}",
        end_ts=f"00:{end:05.2f}",hook_score=score/10,cliffhanger_score=score/10,
        standalone_score=score/10,emotional_intensity=score/10,viewer_curiosity=score/10,
        total_score=score,evidence="screenplay evidence")

def test_camera_only_action_is_not_hook():
    s=Scene(1,"INT",0,30,actions=[ActionLine("Medium shot on the heroine standing silently.",0,2)])
    score,kind,evidence=detect_hook(s,0,30)
    assert score <= 1.5
    assert kind != "Action shock"

def test_camera_direction_with_strong_event_is_hook():
    s=Scene(1,"INT",0,30,actions=[ActionLine("Medium shot as he slaps her.",0,2)])
    score,kind,evidence=detect_hook(s,0,30)
    assert score > 0
    assert kind == "Action shock"

@pytest.mark.parametrize("line", ["I didn't!", "Yes.", "No!", "Okay?"])
def test_punctuation_only_denial_or_agreement_is_not_cliffhanger(line):
    s=Scene(1,"INT",0,30,dialogue=[DialogueLine("A","",line,25,29)])
    score,kind,_=detect_cliffhanger(s,0,30)
    assert score <= 2

def test_question_with_unanswered_signal_is_cliffhanger():
    s=Scene(1,"INT",0,30,dialogue=[DialogueLine("A","","Who is the person behind the door?",25,29)])
    score,_,_=detect_cliffhanger(s,0,30)
    assert score >= 4

def test_threat_with_exclamation_is_cliffhanger():
    s=Scene(1,"INT",0,30,dialogue=[DialogueLine("A","","You will pay for this!",25,29)])
    score,_,_=detect_cliffhanger(s,0,30)
    assert score >= 4

def test_rank_based_tiers_and_episode_cap():
    items=[candidate(i%5+1,(i//5)*100,(i//5)*100+30,100-i) for i in range(35)]
    ranked=rank_candidates(items)
    assert ranked[0].recommendation.endswith("Post First")
    assert ranked[10].recommendation.endswith("Post Later")
    assert ranked[30].recommendation.endswith("Human Review")
    picked=top_n(ranked,10)
    counts={}
    for c in picked: counts[c.episode]=counts.get(c.episode,0)+1
    assert all(n<=2 for n in counts.values())

def test_overlap_dedupe_keeps_higher_score():
    low=candidate(1,0,40,60); high=candidate(1,10,50,80)
    ranked=rank_candidates([low,high])
    assert len(ranked)==1 and ranked[0] is high
    assert overlap_ratio(high,low)>0.4

def test_overlap_dedupe_does_not_merge_different_episodes():
    a=candidate(1,0,40,60); b=candidate(2,0,40,50)
    assert len(rank_candidates([a,b]))==2

def test_gold_compare_hit_at_one_and_three_and_miss():
    gold=[{"episode":1,"start":10.0,"end":30.0,"notes":""},
          {"episode":2,"start":50.0,"end":70.0,"notes":""}]
    preds=[{"episode":1,"start":10.0,"end":30.0,"rank":1,"valid":True},
           {"episode":2,"start":0.0,"end":10.0,"rank":1,"valid":False},
           {"episode":2,"start":49.0,"end":65.0,"rank":2,"valid":True}]
    result=evaluate(gold,preds)
    assert result["hit1"]==0.5
    assert result["hit3"]==1.0
    assert len(result["misses"])==0


def test_run_all_mock_creates_separate_outputs_and_clears_old_files(tmp_path):
    import os, re, subprocess
    project=Path(__file__).resolve().parents[1]
    sample=(project/"sample_data"/"sample_screenplay.txt").read_text(encoding="utf-8")
    folder=tmp_path/"new_show"; folder.mkdir()
    blocks=[b for b in re.split(r"(?=EPISODE NUMBER:)",sample) if "EPISODE NUMBER:" in b]
    for i,block in enumerate(blocks,1):
        (folder/f"episode_{i:02}.txt").write_text(block,encoding="utf-8")
    out=tmp_path/"outputs"; out.mkdir(); (out/"stale.csv").write_text("stale", encoding="utf-8")
    result=subprocess.run([sys.executable,"run_all.py",str(folder),"--mock","--out",str(out)],
                          cwd=project,capture_output=True,text=True)
    assert result.returncode != 0
    assert "unrelated files" in result.stderr
    assert (out/"stale.csv").read_text(encoding="utf-8") == "stale"

def test_shared_hook_window_configuration():
    import config, llm_selector
    assert llm_selector.HOOK_WINDOW_S == config.HOOK_WINDOW_SECONDS == 3.0


def test_wait_with_exclamation_is_not_enough_for_cliffhanger():
    s=Scene(1,"INT",0,30,dialogue=[DialogueLine("A","","Wait!",25,29)])
    score,_,_=detect_cliffhanger(s,0,30)
    assert score <= 2

def test_cut_to_black_without_unresolved_action_is_not_enough():
    s=Scene(1,"INT",0,30,actions=[ActionLine("The scene cuts to black.",25,29)])
    score,_,_=detect_cliffhanger(s,0,30)
    assert score <= 2


def test_gold_compare_reads_tiny_csv_and_json_files(tmp_path):
    from evaluation.compare import load_gold, load_preds, evaluate
    gold_file=tmp_path/"gold.csv"
    gold_file.write_text("episode,start_ts,end_ts,notes\n1,00:10.00,00:30.00,editor pick\n", encoding="utf-8")
    pred_file=tmp_path/"pred.json"
    pred_file.write_text(json.dumps([{"episode":1,"rank":1,"start_ts":"00:11.00",
        "end_ts":"00:29.00","quality_flags":[]}]), encoding="utf-8")
    gold=load_gold(gold_file); preds=load_preds(pred_file)
    result=evaluate(gold,preds)
    assert result["hit1"]==1.0 and result["hit3"]==1.0
    assert result["validation_share"]==1.0


def test_quality_gate_overrides_highest_ranked_candidate():
    weak = candidate(1, 0, 30, 100)
    weak.cliffhanger_score = 0
    clean = candidate(2, 0, 30, 90)
    ranked = rank_candidates([weak, clean])
    assert ranked[0] is weak
    assert "Human Review" in weak.recommendation
    assert clean.recommendation.endswith("Post First")


def test_post_first_episode_cap_moves_extra_clean_clips_to_post_later(monkeypatch):
    import config
    monkeypatch.setattr(config, "POST_FIRST_COUNT", 10)
    monkeypatch.setattr(config, "POST_LATER_COUNT", 20)
    monkeypatch.setattr(config, "TOP_LIST_EPISODE_CAP", 2)
    items = [candidate(1, i*100, i*100+30, 100-i) for i in range(4)]
    items += [candidate(i, 1000+i*100, 1030+i*100, 90-i) for i in range(2, 10)]
    ranked = rank_candidates(items)
    first = [c for c in ranked if c.recommendation.endswith("Post First")]
    later = [c for c in ranked if c.recommendation.endswith("Post Later")]
    assert sum(c.episode == 1 for c in first) == 2
    assert any(c.episode == 1 for c in later)


def test_no_api_key_real_run_exits_before_processing(tmp_path):
    import os
    project = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ("LLM_API_KEY", "OPENAI_API_KEY"):
        env.pop(key, None)
    result = subprocess.run([sys.executable, "run_all.py", str(tmp_path)], cwd=project,
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "No LLM_API_KEY set. Set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL, or use --mock." in result.stderr


def test_run_all_refuses_to_delete_unrelated_output_file(tmp_path):
    import re
    project = Path(__file__).resolve().parents[1]
    sample = (project / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    blocks = [b for b in re.split(r"(?=EPISODE NUMBER:)", sample) if "EPISODE NUMBER:" in b]
    folder = tmp_path / "show"; folder.mkdir()
    (folder / "episode_1.txt").write_text(blocks[0], encoding="utf-8")
    out = tmp_path / "outputs"; out.mkdir()
    precious = out / "do_not_delete.txt"; precious.write_text("keep me", encoding="utf-8")
    result = subprocess.run([sys.executable, "run_all.py", str(folder), "--mock", "--out", str(out)],
                            cwd=project, capture_output=True, text=True)
    assert result.returncode != 0
    assert precious.read_text(encoding="utf-8") == "keep me"
    assert "unrelated files" in result.stderr


def test_run_summary_contains_rejection_reason_counts(tmp_path):
    import re
    project = Path(__file__).resolve().parents[1]
    sample = (project / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    block = next(b for b in re.split(r"(?=EPISODE NUMBER:)", sample) if "EPISODE NUMBER:" in b)
    folder = tmp_path / "show"; folder.mkdir(); (folder / "episode_1.txt").write_text(block, encoding="utf-8")
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, "run_all.py", str(folder), "--mock", "--out", str(out)],
                            cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary["llm"]["rejection_reasons"], dict)


def test_default_config_excludes_show_specific_phrases():
    import config
    all_words = " ".join(str(v) for k, v in vars(config).items() if k.endswith("KEYWORDS") or k in ("CLIFFHANGER_DIALOGUE", "CLIFFHANGER_VISUAL")).lower()
    for phrase in ("you are disowned", "dead to this family", "he loves me now",
                   "take your hands off her", "remove her", "someone planted this"):
        assert phrase not in all_words


def test_label_helper_prints_numbered_timestamp_lines_without_writing(tmp_path):
    from evaluation.label_helper import format_episode
    project = Path(__file__).resolve().parents[1]
    sample = (project / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    import re
    block = next(b for b in re.split(r"(?=EPISODE NUMBER:)", sample) if "EPISODE NUMBER:" in b)
    folder = tmp_path / "show"; folder.mkdir(); (folder / "episode_1.txt").write_text(block, encoding="utf-8")
    text = format_episode(folder, 1)
    assert "EPISODE 1" in text
    assert "L001 [" in text


def test_report_labels_mock_and_includes_high_low_hook_type(tmp_path):
    from evaluation.make_report import main
    import sys
    outdir=tmp_path/"outputs"; outdir.mkdir()
    (outdir/"run_summary.json").write_text(json.dumps({"mock":True,"model":"MOCK","episodes":1,"tokens":{},"llm":{"rejection_reasons":{}}}), encoding="utf-8")
    candidate_row={"episode":1,"start_ts":"00:00.00","end_ts":"00:30.00","total_score":40,
        "recommendation":"👁️ Needs Human Review","quality_flags":["Weak hook"],
        "hook":{"score":1,"type":"Dialogue","evidence":"hello"},
        "cliffhanger":{"score":1,"evidence":"yes"}}
    for name in ("rule_based_promos.json","llm_promos.json"):
        (outdir/name).write_text(json.dumps([candidate_row]), encoding="utf-8")
    (outdir/"llm_rejected.json").write_text("[]", encoding="utf-8")
    target=tmp_path/"EVIDENCE.md"
    old=sys.argv; sys.argv=["make_report.py","--outputs",str(outdir),"--out",str(target)]
    try: main()
    finally: sys.argv=old
    report=target.read_text(encoding="utf-8")
    assert "MOCK RUN" in report and "10 highest-scoring" in report and "Hook-type distribution" in report
    assert "Score histograms" in report and "Severe weakness shares" in report and "Hook:" in report


def test_run_all_preserves_outputs_gitkeep(tmp_path):
    import re
    project = Path(__file__).resolve().parents[1]
    sample = (project / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    block = next(b for b in re.split(r"(?=EPISODE NUMBER:)", sample) if "EPISODE NUMBER:" in b)
    folder = tmp_path / "show"; folder.mkdir(); (folder / "episode_1.txt").write_text(block, encoding="utf-8")
    out = tmp_path / "out"; out.mkdir(); (out / ".gitkeep").write_text("", encoding="utf-8")
    result = subprocess.run([sys.executable, "run_all.py", str(folder), "--mock", "--out", str(out)],
                            cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (out / ".gitkeep").exists()


def test_cliffhanger_window_uses_central_config():
    import config, llm_selector
    assert llm_selector.CLIFF_WINDOW_S == config.CLIFFHANGER_WINDOW_SECONDS


def test_analysis_source_has_no_removed_show_plot_phrases():
    root = Path(__file__).resolve().parents[1] / "analysis"
    source = " ".join(p.read_text(encoding="utf-8").lower() for p in root.glob("*.py"))
    for phrase in ("you are disowned", "dead to this family", "he loves me now",
                   "take your hands off her", "remove her", "someone planted this",
                   "rightful duke", "blackthorn", "witchcraft"):
        assert phrase not in source


def test_post_first_quota_slots_skip_episode_cap_overflow():
    import config
    old = (config.POST_FIRST_COUNT, config.POST_LATER_COUNT, config.TOP_LIST_EPISODE_CAP)
    config.POST_FIRST_COUNT, config.POST_LATER_COUNT, config.TOP_LIST_EPISODE_CAP = 10, 20, 2
    try:
        items = [candidate(1, i * 100, i * 100 + 30, 200 - i) for i in range(6)]
        items += [candidate(ep, 1000 + ep * 100, 1030 + ep * 100, 100 - ep)
                  for ep in range(2, 12)]
        ranked = rank_candidates(items)
        first = [c for c in ranked if c.recommendation.endswith("Post First")]
        assert len(first) == 10
        counts = {}
        for c in first:
            counts[c.episode] = counts.get(c.episode, 0) + 1
        assert max(counts.values()) <= 2
    finally:
        config.POST_FIRST_COUNT, config.POST_LATER_COUNT, config.TOP_LIST_EPISODE_CAP = old


def test_shared_matcher_uses_word_boundaries_for_ambiguous_terms():
    import config
    from analysis.text_match import kw_hit
    keywords = config.HOOK_ACTION_KEYWORDS
    assert kw_hit("fireplace", keywords) == 0
    assert kw_hit("waterfall", keywords) == 0
    assert kw_hit("in charge of the accounts", keywords) == 0
    assert "charge" not in keywords and "fall" not in keywords


def test_shared_matcher_matches_inflected_event_verbs():
    import config
    from analysis.text_match import kw_hit
    assert kw_hit("He slaps her", config.HOOK_ACTION_KEYWORDS) >= 1
    assert kw_hit("she collapses", config.HOOK_ACTION_KEYWORDS) >= 1


def test_shared_matcher_supports_keywords_ending_in_punctuation():
    from analysis.text_match import kw_hit
    assert kw_hit("The answer is who?", ["who?"]) == 1


def test_run_all_mock_writes_all_six_outputs_and_summary_fields(tmp_path):
    import re
    project = Path(__file__).resolve().parents[1]
    sample = (project / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    block = next(b for b in re.split(r"(?=EPISODE NUMBER:)", sample) if "EPISODE NUMBER:" in b)
    folder = tmp_path / "sample_show"
    folder.mkdir()
    (folder / "episode_1.txt").write_text(block, encoding="utf-8")
    out = tmp_path / "outputs"
    result = subprocess.run([sys.executable, "run_all.py", str(folder), "--mock", "--out", str(out)],
                            cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = {"rule_based_promos.csv", "rule_based_promos.json", "llm_promos.csv",
                "llm_promos.json", "llm_rejected.json", "run_summary.json"}
    assert expected.issubset({p.name for p in out.iterdir()})
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["mock"] is True
    assert summary["episodes"] == 1
    assert "rule_based" in summary and "llm" in summary and "tokens" in summary

# Final export/snap regression coverage.
def _sample_episode():
    from parser.txt_parser import parse_txt_content
    raw = (Path(__file__).resolve().parents[1] / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    import re
    block = next(b for b in re.split(r"(?=EPISODE NUMBER:)", raw) if "EPISODE NUMBER: Episode 1" in b)
    return parse_txt_content(block, "episode_1.txt")


def _proposal_for_lines(lines, start_idx, end_idx, hook_idx, cliff_idx):
    return {"promos": [{"start_line": lines[start_idx]["id"], "end_line": lines[end_idx]["id"],
        "hook_line": lines[hook_idx]["id"], "cliffhanger_line": lines[cliff_idx]["id"],
        "hook": "hook", "cliffhanger": "unresolved", "why": "test",
        "scores": {"hook": 8, "cliffhanger": 8, "standalone": 8, "emotion": 6, "curiosity": 8}}]}


def test_single_segment_json_csv_exports_and_legacy_timestamps():
    from export.json_export import candidates_to_json_bytes
    from export.csv_export import candidates_to_csv_bytes
    import csv, io
    c = candidate(12, 14.5, 48.0, 80)
    obj = json.loads(candidates_to_json_bytes([c]))[0]
    assert obj["segments"] == [{"episode": 12, "start_ts": c.start_ts, "end_ts": c.end_ts}]
    assert obj["start_ts"] == c.start_ts and obj["end_ts"] == c.end_ts
    row = next(csv.DictReader(io.StringIO(candidates_to_csv_bytes([c]).decode())))
    assert row["segments"] == f"EP12 {c.start_ts}–{c.end_ts}"
    assert row["start_ts"] == c.start_ts and row["end_ts"] == c.end_ts


def test_two_segment_exports_preserve_order_and_csv_separator():
    from export.json_export import candidates_to_json_bytes
    from export.csv_export import candidates_to_csv_bytes
    import csv, io
    c = candidate(12, 14.5, 48.0, 80)
    c.segments = [
        {"episode": 12, "start_ts": "00:14.50", "end_ts": "00:28.00"},
        {"episode": 13, "start_ts": "00:02.00", "end_ts": "00:20.00"},
    ]
    obj = json.loads(candidates_to_json_bytes([c]))[0]
    assert [x["episode"] for x in obj["segments"]] == [12, 13]
    row = next(csv.DictReader(io.StringIO(candidates_to_csv_bytes([c]).decode())))
    assert row["segments"] == "EP12 00:14.50–00:28.00 | EP13 00:02.00–00:20.00"


def test_snap_off_rejects_late_hook_exactly_as_before():
    from llm_selector import build_lines, select_promos
    ep = _sample_episode(); lines = build_lines(ep)
    end_idx = next(i for i, line in enumerate(lines) if line["end"] - lines[0]["start"] >= 30)
    hook_idx = next(i for i, line in enumerate(lines) if line["start"] - lines[0]["start"] > 3)
    response = _proposal_for_lines(lines, 0, end_idx, hook_idx, end_idx)
    got, rejected = select_promos(ep, lambda *_: json.dumps(response), max_promos=1)
    assert not got
    assert rejected[0]["reason"] == "hook line is not within first 3s of clip"


def test_snap_on_trims_to_real_hook_timestamp_and_marks_adjusted():
    from llm_selector import build_lines, select_promos
    ep = _sample_episode(); lines = build_lines(ep)
    end_idx = next(i for i, line in enumerate(lines) if line["end"] - lines[0]["start"] >= 40)
    hook_idx = next(i for i, line in enumerate(lines) if line["start"] - lines[0]["start"] > 3 and line["start"] - lines[0]["start"] < 10)
    response = _proposal_for_lines(lines, 0, end_idx, hook_idx, end_idx)
    got, rejected = select_promos(ep, lambda *_: json.dumps(response), max_promos=1, snap=True)
    assert got and not rejected
    assert got[0].start == lines[hook_idx]["start"]
    assert got[0].start_ts
    assert got[0].adjusted is True


def test_snap_rejects_when_trimmed_duration_is_too_short():
    from llm_selector import build_lines, select_promos
    ep = _sample_episode(); lines = build_lines(ep)
    start_idx = 0
    end_idx = next(i for i, line in enumerate(lines) if line["end"] - lines[start_idx]["start"] >= 25)
    hook_idx = next(i for i, line in enumerate(lines) if 8 < line["start"] - lines[start_idx]["start"] < 12)
    cliff_idx = next(i for i, line in enumerate(lines) if 12 < line["end"] - lines[start_idx]["start"] < 18)
    response = _proposal_for_lines(lines, start_idx, end_idx, hook_idx, cliff_idx)
    got, rejected = select_promos(ep, lambda *_: json.dumps(response), max_promos=1, snap=True)
    assert not got
    assert any(item["reason"] == "snap would break duration" for item in rejected)


def test_hook_after_cliffhanger_is_rejected_even_with_snap():
    from llm_selector import build_lines, select_promos
    ep = _sample_episode(); lines = build_lines(ep)
    end_idx = next(i for i, line in enumerate(lines) if line["end"] - lines[0]["start"] >= 30)
    hook_idx = end_idx
    cliff_idx = 0
    response = _proposal_for_lines(lines, 0, end_idx, hook_idx, cliff_idx)
    got, rejected = select_promos(ep, lambda *_: json.dumps(response), max_promos=1, snap=True)
    assert not got
    assert rejected[0]["reason"] == "hook line is after cliffhanger line"


def test_mock_run_summary_has_rejection_metrics_and_snap_flag(tmp_path):
    import re, subprocess, sys
    project = Path(__file__).resolve().parents[1]
    raw = (project / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    blocks = [b for b in re.split(r"(?=EPISODE NUMBER:)", raw) if "EPISODE NUMBER:" in b]
    folder = tmp_path / "multi"; folder.mkdir()
    for i, block in enumerate(blocks[:2], 1): (folder / f"ep{i}.txt").write_text(block, encoding="utf-8")
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, "run_all.py", str(folder), "--mock", "--snap", "--out", str(out)], cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    llm = summary["llm"]
    assert llm["proposals_total"] == llm["proposals_rejected"] + llm["candidates_before_dedupe"]
    assert llm["rejection_rate"] == (llm["proposals_rejected"] / llm["proposals_total"] if llm["proposals_total"] else 0)
    assert llm["snap_enabled"] is True


def test_kw_hit_matches_regular_past_tense_and_gerund_inflections():
    from analysis.text_match import kw_hit
    for word, base in [("slapped", "slap"), ("shoved", "shove"),
                       ("collapsed", "collapse"), ("grabbed", "grab"),
                       ("exploded", "explode")]:
        assert kw_hit(f"She {word} him.", [base]) == 1


def test_kw_hit_does_not_match_substrings_inside_other_words():
    from analysis.text_match import kw_hit
    assert kw_hit("The fireplace burned.", ["fire"]) == 0
    assert kw_hit("A slapstick comedy.", ["slap"]) == 0


def test_apply_llm_overrides_and_restore_values():
    import llm_client
    from run_all import apply_llm_overrides
    old_model, old_url = llm_client.MODEL, llm_client.BASE_URL
    try:
        apply_llm_overrides("test-model", "https://example.test/v1")
        assert llm_client.MODEL == "test-model"
        assert llm_client.BASE_URL == "https://example.test/v1"
    finally:
        apply_llm_overrides(old_model, old_url)
    assert llm_client.MODEL == old_model
    assert llm_client.BASE_URL == old_url


def test_run_all_warns_about_unparsed_files_and_summarizes_input(tmp_path):
    import re
    project = Path(__file__).resolve().parents[1]
    sample = (project / "sample_data" / "sample_screenplay.txt").read_text(encoding="utf-8")
    block = next(b for b in re.split(r"(?=EPISODE NUMBER:)", sample) if "EPISODE NUMBER:" in b)
    folder = tmp_path / "show"; folder.mkdir()
    (folder / "episode_1.txt").write_text(block, encoding="utf-8")
    (folder / "junk.txt").write_text("not a screenplay", encoding="utf-8")
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, "run_all.py", str(folder), "--mock", "--out", str(out)],
                            cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "WARNING: 1 of 2 .txt files could not be parsed" in result.stderr
    assert "junk.txt:" in result.stderr          # the reason is reported per file
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["input"]["files_found"] == 2
    assert summary["input"]["episodes_parsed"] == 1
    assert summary["input"]["unparsed_files"] == ["junk.txt"]
    assert "junk.txt" in summary["input"]["unparsed_reasons"]


def test_run_all_errors_when_zero_episodes_parse(tmp_path):
    project = Path(__file__).resolve().parents[1]
    (tmp_path / "junk.txt").write_text("junk", encoding="utf-8")
    result = subprocess.run([sys.executable, "run_all.py", str(tmp_path), "--mock"],
                            cwd=project, capture_output=True, text=True)
    assert result.returncode == 2
    assert "No episodes could be parsed. Expected format: see README 'Input format'." in result.stderr


def test_snap_evidence_and_scene_range_only_use_lines_inside_final_clip():
    from llm_selector import _validate, build_lines
    from parser.txt_parser import Episode, Scene, DialogueLine
    ep = Episode(number=9, filename="ep9.txt", logline="", continuity="", scenes=[
        Scene(1, "INT", 0, 50, dialogue=[
            DialogueLine("A", "", "Outside before", 0, 2),
            DialogueLine("A", "", "Hook inside", 5, 8),
            DialogueLine("B", "", "Cliff inside", 35, 40),
            DialogueLine("B", "", "Outside after", 45, 48),
        ])
    ])
    lines = build_lines(ep); idx = {line["id"]: i for i, line in enumerate(lines)}
    proposal = {"start_line": lines[0]["id"], "end_line": lines[3]["id"],
                "hook_line": lines[1]["id"], "cliffhanger_line": lines[2]["id"],
                "scores": {"hook": 5, "cliffhanger": 5}}
    candidate, reason = _validate(proposal, ep, lines, idx, 1, snap=True)
    assert candidate is not None, reason
    assert "Outside before" not in candidate.evidence
    assert "Outside after" not in candidate.evidence
    assert candidate.start == 5 and candidate.end == 40
    assert candidate.scene_range == (1, 1)
