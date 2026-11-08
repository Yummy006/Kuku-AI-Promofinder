"""Compare ranked outputs with editor-labelled gold clips.

Gold hit: intersection / gold duration >= 0.50. Supports JSON exports.
"""
import argparse, csv, json
from pathlib import Path

def ts(value):
    if isinstance(value,(int,float)): return float(value)
    bits=str(value).strip().split(":")
    return sum(float(v)*60**i for i,v in enumerate(reversed(bits)))

def load_gold(path):
    rows=[]
    with open(path,encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("episode") or not row.get("start_ts") or not row.get("end_ts"): continue
            rows.append({"episode":int(row["episode"]),"start":ts(row["start_ts"]),
                         "end":ts(row["end_ts"]),"notes":row.get("notes","")})
    return rows

def load_preds(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data,dict): data=data.get("candidates",data.get("promos",[]))
    out=[]
    for x in data:
        start=x.get("start",x.get("start_sec",x.get("start_ts")))
        end=x.get("end",x.get("end_sec",x.get("end_ts")))
        if start is None or end is None: continue
        out.append({"episode":int(x["episode"]),"start":ts(start),"end":ts(end),
                    "rank":int(x.get("rank",99999)),
                    "valid":not bool(x.get("quality_flags",[]))})
    return sorted(out,key=lambda x:(x["rank"],x["episode"],x["start"]))

def evaluate(gold,preds):
    episodes=sorted({g["episode"] for g in gold})
    hits1=hits3=0; overlaps=[]; misses=[]
    for ep in episodes:
        gs=[g for g in gold if g["episode"]==ep]
        ps=[p for p in preds if p["episode"]==ep]
        best_per_gold=[]
        for g in gs:
            gd=max(g["end"]-g["start"],0.001)
            ovs=[]
            for p in ps[:3]:
                inter=max(0,min(g["end"],p["end"])-max(g["start"],p["start"]))
                ovs.append((inter/gd,p))
            best=max((v for v,_ in ovs),default=0)
            overlaps.append(best)
            best_per_gold.append(best)
            if best<0.5:
                misses.append(f"EP{ep}: gold {g['start']:.2f}-{g['end']:.2f}s; top predictions: "+
                              (", ".join(f"{p['start']:.2f}-{p['end']:.2f}s" for p in ps[:3]) or "none"))
        if any(max((max(0,min(g["end"],p["end"])-max(g["start"],p["start"])) /
                    max(g["end"]-g["start"],0.001) for p in ps[:1]),default=0)>=0.5 for g in gs):
            hits1+=1
        if any(max((max(0,min(g["end"],p["end"])-max(g["start"],p["start"])) /
                    max(g["end"]-g["start"],0.001) for p in ps[:3]),default=0)>=0.5 for g in gs):
            hits3+=1
    n=max(len(episodes),1)
    return {"hit1":hits1/n,"hit3":hits3/n,"avg_overlap":sum(overlaps)/len(overlaps) if overlaps else 0,
            "validation_share":sum(p["valid"] for p in preds)/len(preds) if preds else 0,
            "misses":misses,"episodes":len(episodes),"gold_clips":len(gold),"predictions":len(preds)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold",default="evaluation/gold_template.csv")
    ap.add_argument("--rule",default="outputs/rule_based_promos.json")
    ap.add_argument("--llm",default="outputs/llm_promos.json")
    ap.add_argument("--out",default="evaluation/results.md")
    a=ap.parse_args()
    gold=load_gold(a.gold)
    methods={}
    for name,path in (("Rule-based",a.rule),("LLM",a.llm)):
        if Path(path).exists(): methods[name]=evaluate(gold,load_preds(path))
    lines=["# PromoFinder evaluation results","",
           f"Gold labels: {len(gold)} clips across {len({g['episode'] for g in gold})} episodes.","",
           "| Method | Predictions | Hit@1 / episode | Hit@3 / episode | Average best overlap | Validation pass share |",
           "|---|---:|---:|---:|---:|---:|"]
    for name,m in methods.items():
        lines.append(f"| {name} | {m['predictions']} | {m['hit1']:.1%} | {m['hit3']:.1%} | {m['avg_overlap']:.1%} | {m['validation_share']:.1%} |")
    for name,m in methods.items():
        lines += ["",f"## Misses — {name}"]
        lines += [f"- {x}" for x in m["misses"]] or ["- No misses against the supplied gold labels."]
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))
if __name__=="__main__": main()
