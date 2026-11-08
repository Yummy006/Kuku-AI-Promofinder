"""Offline smoke test on synthetic sample screenplay. Run: python run_pipeline_test.py"""
import os
from pathlib import Path
from parser.txt_parser import parse_txt_content
from analysis.screenplay_analyzer import analyze_all_episodes
from scoring.promo_scorer import score_all
from scoring.ranking import rank_candidates
from evaluation.quality_checker import run_quality_check

def main():
    raw=(Path(__file__).parent/"sample_data"/"sample_screenplay.txt").read_text(encoding="utf-8")
    import re
    blocks=re.split(r"(?=EPISODE NUMBER:)",raw)
    episodes=[e for i,b in enumerate(blocks) if "EPISODE NUMBER:" in b
              for e in [parse_txt_content(b,f"synthetic_{i}.txt")] if e]
    print(f"Parsed {len(episodes)} synthetic episodes")
    candidates=analyze_all_episodes(episodes)
    score_all(candidates); ranked=rank_candidates(candidates)
    report=run_quality_check(ranked)
    print(f"Candidates: {len(ranked)} | validation pass rate: {report['pass_rate']}%")
    print("Synthetic smoke test completed. This is not a quality benchmark.")

if __name__=="__main__": main()
