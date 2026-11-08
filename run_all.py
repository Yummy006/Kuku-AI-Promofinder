"""Run rule-based and LLM selection on the same episode folder.

Usage: python3 run_all.py "/path/to/show" [--limit N] [--mock]
"""
import argparse, json, os, time, sys, collections
from pathlib import Path
from parser.txt_parser import parse_txt_file_ex
from analysis.screenplay_analyzer import analyze_all_episodes
from scoring.promo_scorer import score_all
from scoring.ranking import rank_candidates
from evaluation.quality_checker import run_quality_check
from export.csv_export import candidates_to_csv_bytes
from export.json_export import candidates_to_json_bytes
import config, llm_client
from llm_selector import select_promos

# Stop calling the model after this many episodes in a row could not reach it.
MAX_CONSECUTIVE_LLM_FAILURES = 3

# Files this runner owns inside --out. promo_candidates.json / quality_report.json are
# written by the Streamlit app into the same default folder, so they are managed too.
KNOWN_OUTPUTS = {"rule_based_promos.csv", "rule_based_promos.json", "llm_promos.csv",
                 "llm_promos.json", "llm_rejected.json", "run_summary.json",
                 "promo_candidates.json", "quality_report.json", ".gitkeep"}


def mock_client(system, user):
    # Deterministic proposal based only on IDs/timestamps supplied to model.
    import re
    rows=re.findall(r"^(L\d+) \[(\d+):([\d.]+)-(\d+):([\d.]+)\]",user,re.M)
    if not rows: return '{"promos":[]}'
    sec=lambda row:int(row[1])*60+float(row[2])
    endrow=next((x for x in rows if int(x[3])*60+float(x[4])-sec(rows[0])>=20),None)
    if not endrow: return '{"promos":[]}'
    return json.dumps({"promos":[{"start_line":rows[0][0],"end_line":endrow[0],
        "hook_line":rows[0][0],"cliffhanger_line":endrow[0],
        "hook":"offline mock","cliffhanger":"offline mock","why":"Mock wiring test only",
        "scores":{"hook":5,"cliffhanger":5,"standalone":5,"emotion":5,"curiosity":5}}]})

def apply_llm_overrides(model=None, base_url=None):
    """Apply per-run model endpoint overrides without exposing credentials."""
    if model is not None:
        llm_client.MODEL = model
    if base_url is not None:
        llm_client.BASE_URL = base_url


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--limit",type=int)
    ap.add_argument("--mock",action="store_true")
    ap.add_argument("--out",default="outputs")
    ap.add_argument("--per-episode",type=int,default=3)
    ap.add_argument("--snap", action="store_true", default=os.getenv("PROMO_SNAP", "0").lower() in ("1", "true", "yes"))
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    args=ap.parse_args()
    apply_llm_overrides(args.model, args.base_url)
    # Preserve the existing fast credential check for non-mock runs.
    if not args.mock and not llm_client.has_credentials():
        print("No LLM_API_KEY set. Set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL, or use --mock.", file=sys.stderr)
        return 2
    folder=Path(args.folder).expanduser().resolve()
    if not folder.is_dir(): ap.error(f"folder does not exist: {folder}")
    txt_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    parsed, unparsed_files, unparsed_reasons, non_utf8_files = [], [], {}, {}
    for path in txt_files:
        file_eps, read_error = parse_txt_file_ex(str(path))   # a file may hold several episodes
        usable = [ep for ep in file_eps if ep.scene_count > 0]
        parsed.extend(usable)
        if not usable:
            unparsed_files.append(path.name)
            unparsed_reasons[path.name] = read_error or (
                "; ".join(file_eps[0].parse_errors) if file_eps and file_eps[0].parse_errors
                else "no scenes found (check the input format in the README)")
        elif usable[0].encoding not in ("utf-8", "utf-8-sig"):
            non_utf8_files[path.name] = usable[0].encoding
    episodes = sorted(parsed, key=lambda e: e.number)
    if unparsed_files:
        print(f"WARNING: {len(unparsed_files)} of {len(txt_files)} .txt files could not be parsed:", file=sys.stderr)
        for name in unparsed_files:
            print(f"  - {name}: {unparsed_reasons[name]}", file=sys.stderr)
    if non_utf8_files:
        listing = ", ".join(f"{n} ({enc})" for n, enc in non_utf8_files.items())
        print(f"WARNING: {len(non_utf8_files)} file(s) were not UTF-8 and were decoded with a fallback: {listing}. "
              "Check special characters, or re-save as UTF-8.", file=sys.stderr)
    if not episodes:
        print("No episodes could be parsed. Expected format: see README 'Input format'.", file=sys.stderr)
        return 2
    if args.limit is not None: episodes=episodes[:max(0,args.limit)]
    print(f"Model: MOCK (offline)" if args.mock else
          f"Model: {llm_client.MODEL} | Base URL: {llm_client.BASE_URL} | per-episode: {args.per_episode}")
    out=Path(args.out).expanduser().resolve()
    if out == folder or folder in out.parents or out in folder.parents:
        ap.error("output folder must not be the input folder, an ancestor of it, or inside it")
    known_outputs = KNOWN_OUTPUTS
    if out.exists():
        if not out.is_dir():
            ap.error(f"output path exists and is not a directory: {out}")
        unexpected = [p.name for p in out.iterdir() if p.name not in known_outputs]
        if unexpected:
            ap.error("refusing to clear output directory containing unrelated files: " + ", ".join(sorted(unexpected)))
        for name in known_outputs - {".gitkeep"}:
            target = out / name
            if target.is_file() or target.is_symlink():
                target.unlink()
            elif target.exists():
                ap.error(f"refusing to remove unexpected directory: {target}")
    out.mkdir(parents=True, exist_ok=True)
    start=time.perf_counter()

    rb_raw=analyze_all_episodes(episodes)
    score_all(rb_raw)
    rb_before_dedupe=len(rb_raw)
    rb=rank_candidates(rb_raw); rb_report=run_quality_check(rb)

    client=mock_client if args.mock else llm_client.chat
    llm=[]; rejected=[]
    llm_client.reset_usage()
    llm_abort=None; failed_episodes=0; consecutive_failures=0; episodes_sent=0
    for ep in episodes:
        episodes_sent+=1
        try:
            got,rej=select_promos(ep,client,args.per_episode, snap=args.snap)
        except llm_client.LLMFatalError as exc:       # bad key / model / URL: stop now
            llm_abort=str(exc)
            rejected.append({"episode":ep.number,"reason":str(exc),"fatal":True})
            break
        llm.extend(got); rejected.extend(rej)
        if not got and any(r.get("transport_error") for r in rej):
            failed_episodes+=1; consecutive_failures+=1
        else:
            consecutive_failures=0
        if consecutive_failures>=MAX_CONSECUTIVE_LLM_FAILURES:
            llm_abort=(f"{consecutive_failures} episodes in a row could not reach the model. "
                       f"Last error: {rej[-1].get('reason','unknown')}")
            break
    if not args.mock and not llm_abort and episodes and failed_episodes==len(episodes):
        llm_abort="the model could not be reached for any episode"
    prompt_tokens=0 if args.mock else llm_client.USAGE_TOTAL.get("prompt_tokens",0)
    completion_tokens=0 if args.mock else llm_client.USAGE_TOTAL.get("completion_tokens",0)
    score_all(llm)
    llm_before_dedupe=len(llm)
    llm=rank_candidates(llm); llm_report=run_quality_check(llm)

    (out/"rule_based_promos.csv").write_bytes(candidates_to_csv_bytes(rb))
    (out/"rule_based_promos.json").write_bytes(candidates_to_json_bytes(rb))
    (out/"llm_promos.csv").write_bytes(candidates_to_csv_bytes(llm))
    (out/"llm_promos.json").write_bytes(candidates_to_json_bytes(llm))
    dump(out/"llm_rejected.json",rejected)
    elapsed=round(time.perf_counter()-start,2)
    rejection_reasons = dict(collections.Counter(item.get("reason", "unspecified") for item in rejected))
    proposals_total = llm_before_dedupe + len(rejected)
    proposals_rejected = len(rejected)
    rejection_rate = proposals_rejected / proposals_total if proposals_total else 0.0
    dump(out/"run_summary.json",{
        "model":"MOCK" if args.mock else llm_client.MODEL,
        "mock":args.mock,"folder_name":folder.name,"episodes":len(episodes),
        "input":{"files_found":len(txt_files),"episodes_parsed":len(parsed),"unparsed_files":unparsed_files,
                   "unparsed_reasons":unparsed_reasons,"non_utf8_files":non_utf8_files},
        "rule_based":{"candidates_before_dedupe":rb_before_dedupe,"candidates_kept":len(rb),
                      "candidates_rejected":rb_before_dedupe-len(rb),"quality":rb_report},
        "llm":{"candidates_before_dedupe":llm_before_dedupe,"candidates_kept":len(llm),
               "candidates_rejected":len(rejected)+(llm_before_dedupe-len(llm)),"quality":llm_report,
               "rejection_reasons":rejection_reasons,
               "proposals_total":proposals_total,
               "proposals_rejected":proposals_rejected,
               "rejection_rate":rejection_rate,
               "snap_enabled":bool(args.snap),
               "aborted_reason":llm_abort,
               "episodes_sent_to_llm":episodes_sent,
               "episodes_failed_to_reach_model":failed_episodes},
        "seconds_taken":elapsed,
        "tokens":{"prompt":prompt_tokens,"completion":completion_tokens,"total":prompt_tokens+completion_tokens},
        "note":"Token totals are zero in mock mode; provider must return usage for real token accounting."
    })
    print(f"Episodes: {len(episodes)} | rule candidates: {len(rb)} | LLM candidates: {len(llm)} | rejected: {len(rejected)}")
    if not llm and rejected:
        print(f"All LLM episode proposals were rejected or empty. First rejection: {rejected[0].get('reason', 'unspecified')}")
    if not args.mock:
        print(f"LLM rejection rate: {rejection_rate:.0%} ({proposals_rejected}/{proposals_total})")
    print(f"Elapsed: {elapsed}s | token usage: {prompt_tokens+completion_tokens} | outputs: {out}")
    if llm_abort:
        print(f"ERROR: LLM selection aborted: {llm_abort}. Rule-based outputs were still written; "
              f"LLM outputs are partial or empty. Episodes not sent to the model: {len(episodes)-episodes_sent}.",
              file=sys.stderr)
        return 3
    if failed_episodes:
        print(f"WARNING: the model could not be reached for {failed_episodes} of {len(episodes)} episodes; "
              "see llm_rejected.json.", file=sys.stderr)
    return 0

if __name__=="__main__": raise SystemExit(main())
