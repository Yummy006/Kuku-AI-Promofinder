"""Generate a factual evidence sheet from the latest run_all.py outputs."""
import argparse, json, os
from pathlib import Path


def _score(c, key):
    value = c.get(key, {})
    return value.get("score", 0) if isinstance(value, dict) else 0

def _tier_counts(rows):
    labels = ("✅ Post First", "🕐 Post Later", "👁️ Needs Human Review")
    return {label: sum(str(c.get("recommendation", "")).startswith(label) for c in rows) for label in labels}

def _score_histogram(rows, field):
    labels = ("0-2", "2-4", "4-6", "6-8", "8-10")
    counts = {label: 0 for label in labels}
    for row in rows:
        value = _score(row, field)
        if value < 2: bucket = "0-2"
        elif value < 4: bucket = "2-4"
        elif value < 6: bucket = "4-6"
        elif value < 8: bucket = "6-8"
        else: bucket = "8-10"
        counts[bucket] += 1
    return counts


def _severe_weakness_shares(rows):
    total = len(rows)
    if not total:
        return {"hook < 3": "N/A", "cliffhanger < 2": "N/A", "standalone < 2": "N/A"}
    checks = {
        "hook < 3": lambda c: _score(c, "hook") < 3,
        "cliffhanger < 2": lambda c: _score(c, "cliffhanger") < 2,
        "standalone < 2": lambda c: _score(c, "standalone") < 2,
    }
    return {name: f"{sum(check(c) for c in rows) / total:.1%}" for name, check in checks.items()}

def _hook_distribution(rows):
    counts = {}
    for c in rows:
        hook = c.get("hook", {})
        kind = hook.get("type", "Unknown") if isinstance(hook, dict) else "Unknown"
        counts[kind] = counts.get(kind, 0) + 1
    return counts

def _clip_line(c):
    h, ch = c.get("hook", {}), c.get("cliffhanger", {})
    return (f"- EP{c.get('episode')} {c.get('start_ts')}–{c.get('end_ts')} | score {c.get('total_score', 0)} "
            f"| hook: `{(h.get('evidence', '') if isinstance(h, dict) else '')[:140]}` "
            f"| cliffhanger: `{(ch.get('evidence', '') if isinstance(ch, dict) else '')[:140]}` "
            f"| flags: {'; '.join(c.get('quality_flags', [])) or 'none'}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outputs",default="outputs")
    ap.add_argument("--out",default="docs/EVIDENCE.md")
    ap.add_argument("--input-price",type=float,default=float(os.getenv("PROMO_INPUT_USD_PER_M", "0.15")))
    ap.add_argument("--output-price",type=float,default=float(os.getenv("PROMO_OUTPUT_USD_PER_M", "0.60")))
    a=ap.parse_args(); root=Path(a.outputs)
    summary=json.loads((root/"run_summary.json").read_text(encoding="utf-8"))
    def read(name):
        path=root/name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    methods={"Rule-based":read("rule_based_promos.json"),"LLM":read("llm_promos.json")}
    rejects=read("llm_rejected.json")
    lines=["# PromoFinder AI — Run evidence", ""]
    if summary.get("mock"):
        lines += ["> **MOCK RUN — synthetic/offline wiring evidence only. Do not present these results as real screenplay or model-quality evidence.**", ""]
    lines += [f"- Model: {summary.get('model', 'unknown')}", f"- Episodes: {summary.get('episodes', 0)}",
              f"- Runtime: {summary.get('seconds_taken', 'unknown')} seconds", ""]
    for name, rows in methods.items():
        tiers=_tier_counts(rows)
        passed=sum(not c.get("quality_flags",[]) for c in rows)
        lines += [f"## {name}", f"- Candidates: {len(rows)}", f"- Validation pass rate: {passed/len(rows):.1%}" if rows else "- Validation pass rate: N/A (no candidates)",
                  f"- Tier counts: {json.dumps(tiers, ensure_ascii=False)}", f"- Hook-type distribution: {json.dumps(_hook_distribution(rows), ensure_ascii=False)}",
                  f"- Severe weakness shares (score thresholds): {json.dumps(_severe_weakness_shares(rows), ensure_ascii=False)}", "",
                  "### Score histograms (0–2, 2–4, 4–6, 6–8, 8–10)"]
        for field, label in (("hook", "Hook"), ("cliffhanger", "Cliffhanger"), ("standalone", "Standalone")):
            lines.append(f"- {label}: {json.dumps(_score_histogram(rows, field))}")
        lines += ["", f"### 10 lowest-scoring {name} clips"]
        for c in sorted(rows,key=lambda x:x.get("total_score",0))[:10]: lines.append(_clip_line(c))
        lines += ["",f"### 10 highest-scoring {name} clips"]
        for c in sorted(rows,key=lambda x:x.get("total_score",0),reverse=True)[:10]: lines.append(_clip_line(c))
        lines.append("")
    reasons=summary.get("llm",{}).get("rejection_reasons",{})
    if not reasons:
        for item in rejects:
            reason=item.get("reason","unspecified"); reasons[reason]=reasons.get(reason,0)+1
    lines += ["## LLM rejection reasons", ""]
    lines += [f"- {reason}: {count}" for reason,count in sorted(reasons.items(),key=lambda x:-x[1])] or ["- No rejection reasons recorded."]
    tok=summary.get("tokens",{}); total=int(tok.get("total",0) or 0)
    lines += ["", "## Cost", ""]
    if summary.get("mock") or total <= 0:
        lines.append("- Not calculated: this run is mock or has no provider-reported real token usage.")
    else:
        prompt=int(tok.get("prompt",0) or 0); completion=int(tok.get("completion",0) or 0)
        eps=max(1,int(summary.get("episodes",0) or 0))
        cost=(prompt*a.input_price+completion*a.output_price)/1_000_000
        lines += [f"- Provider-reported tokens: {prompt} prompt, {completion} completion, {total} total.",
                  f"- Estimated cost at configured rates (${a.input_price:.4f}/1M input, ${a.output_price:.4f}/1M output): ${cost:.6f} for this run; ${cost/eps:.6f}/episode; ${cost/eps*1000:.4f}/1,000 episodes.",
                  "- Rates are user-configurable estimates, not verified provider invoices."]
    lines += ["", "## Interpretation", "", "Outputs from the two methods are reported separately; they may refer to overlapping story moments. Scores are heuristic. Automated validation is not editor approval."]
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Wrote {out}")
if __name__=="__main__": main()
