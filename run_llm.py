"""
run_llm.py  --  full-pack runner
================================
    python3 run_llm.py "path/to/episodes_folder"            # real model
    python3 run_llm.py "path/to/episodes_folder" --limit 5  # try 5 episodes first
    python3 run_llm.py sample_data/sample_screenplay.txt --mock   # offline wiring test

Writes outputs/llm_promos.csv, outputs/llm_promos.json, outputs/llm_rejected.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_client
from llm_selector import select_promos
from parser.txt_parser import parse_directory, parse_txt_file_ex
from scoring.promo_scorer import score_all
from scoring.ranking import rank_candidates
from evaluation.quality_checker import run_quality_check
from export.csv_export import candidates_to_csv_bytes
from export.json_export import candidates_to_json_bytes


def mock_client(system, user):
    """Offline stand-in: proposes first line -> first line that makes >=25 s."""
    import re
    rows = re.findall(r"^(L\d+) \[(\d+):([\d.]+)-(\d+):([\d.]+)\]", user, re.M)
    t = lambda m, s: int(m) * 60 + float(s)
    first = rows[0]
    end = next((r for r in rows if t(r[3], r[4]) - t(first[1], first[2]) >= 25), rows[-1])
    return json.dumps({"promos": [{"start_line": first[0], "end_line": end[0],
        "hook_line": first[0], "cliffhanger_line": end[0], "hook": "mock", "cliffhanger": "mock",
        "why": "mock run", "scores": {"hook": 6, "cliffhanger": 6, "standalone": 6, "emotion": 6, "curiosity": 6}}]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--limit", type=int); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--per-episode", type=int, default=3); ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    if os.path.isdir(a.path):
        episodes = parse_directory(a.path)
    else:   # a single file; may hold several episodes (like the sample file)
        episodes, read_error = parse_txt_file_ex(a.path)
        if read_error:
            sys.exit(f"Cannot read {a.path}: {read_error}")
        episodes = [e for e in episodes if e.scene_count > 0]
    if not a.mock and not llm_client.has_credentials():
        sys.exit("No LLM_API_KEY set. Set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL, or use --mock.")
    episodes = sorted(episodes, key=lambda e: e.number)[: a.limit]
    client = mock_client if a.mock else llm_client.chat
    print(f"Episodes: {len(episodes)} | model: {'MOCK' if a.mock else llm_client.MODEL}")

    cands, rejected, fatal = [], [], False
    for ep in episodes:
        try:
            got, rej = select_promos(ep, client, a.per_episode)
        except llm_client.LLMFatalError as exc:       # bad key / model / URL: stop, keep what we have
            print(f"ERROR: {exc}", file=sys.stderr)
            rejected.append({"episode": ep.number, "reason": str(exc), "fatal": True})
            fatal = True
            break
        cands += got; rejected += rej
        print(f"  EP{ep.number}: {len(got)} kept, {len(rej)} rejected")

    score_all(cands)
    cands = rank_candidates(cands)        # dedupe + tiers; keep the returned list
    run_quality_check(cands)              # keeps the selector's own flags and adds checks

    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/llm_promos.csv", "wb").write(candidates_to_csv_bytes(cands))
    open(f"{a.out}/llm_promos.json", "wb").write(candidates_to_json_bytes(cands))
    with open(f"{a.out}/llm_rejected.json", "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(cands)} promos, {len(rejected)} rejections -> {a.out}/")
    for c in cands[:10]:
        print(f"  #{c.rank} EP{c.episode} {c.start_ts}-{c.end_ts} score={c.total_score} {c.recommendation}")
    return 3 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
