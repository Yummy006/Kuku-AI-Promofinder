"""Explainable validation checks for generated promotional candidates."""
from typing import List, Dict, Any
from analysis.screenplay_analyzer import PromoCandidate
import config
from analysis.overlap import is_duplicate

def check_candidate(candidate: PromoCandidate) -> List[str]:
    flags=[]
    scores={"hook":candidate.hook_score,"cliffhanger":candidate.cliffhanger_score,
            "standalone":candidate.standalone_score,"emotion":candidate.emotional_intensity,
            "curiosity":candidate.viewer_curiosity}
    if candidate.hook_score < 3:
        flags.append(f"Weak hook (score {candidate.hook_score}/10).")
    if candidate.cliffhanger_score < 2:
        flags.append(f"Weak cliffhanger (score {candidate.cliffhanger_score}/10).")
    if candidate.standalone_score < 2:
        flags.append(f"Low standalone context (score {candidate.standalone_score}/10).")
    if candidate.duration < config.MIN_PROMO_DURATION:
        flags.append(f"Too short ({candidate.duration:.1f}s).")
    elif candidate.duration > config.MAX_PROMO_DURATION:
        flags.append(f"Too long ({candidate.duration:.1f}s).")
    if candidate.start < 0 or candidate.end <= candidate.start:
        flags.append("Invalid timestamps.")
    if abs((candidate.end-candidate.start)-candidate.duration)>1:
        flags.append("Timestamp/duration mismatch.")
    if candidate.start==0 and candidate.end==0:
        flags.append("Missing timestamps.")
    for label,value in scores.items():
        if not isinstance(value,(int,float)) or not 0<=value<=10:
            flags.append(f"Invalid {label} score.")
    if not (candidate.hook_evidence or candidate.cliffhanger_evidence or candidate.evidence):
        flags.append("No screenplay evidence.")
    return flags

def flag_duplicates(candidates: List[PromoCandidate]) -> None:
    # Ranking removes duplicates; this function is retained for callers and
    # deliberately reports duplicates if invoked on an unranked raw set.
    for i,c in enumerate(candidates):
        for other in candidates[:i]:
            if is_duplicate(other,c):
                c.quality_flags.append(f"Overlap duplicate of {other.promo_id}.")
                break

def run_quality_check(candidates: List[PromoCandidate]) -> Dict[str, Any]:
    for c in candidates:
        # Preserve upstream LLM validation warnings and add deterministic checks.
        upstream=list(c.quality_flags or [])
        c.quality_flags=list(dict.fromkeys(upstream + check_candidate(c)))
    total=len(candidates); valid=sum(not c.quality_flags for c in candidates)
    errors={}
    for c in candidates:
        for flag in c.quality_flags: errors[flag]=errors.get(flag,0)+1
    return {
        "total_candidates":total,"valid_candidates":valid,"invalid_candidates":total-valid,
        "average_score":round(sum(c.total_score for c in candidates)/total,1) if total else 0.0,
        "common_errors":sorted(errors.items(),key=lambda x:(-x[1],x[0])),
        "pass_rate":round(valid/total*100,1) if total else 0.0,
        "review_required":sum("Human Review" in c.recommendation for c in candidates),
        "tier_distribution":{
            "Post First":sum(c.recommendation.endswith("Post First") for c in candidates),
            "Post Later":sum(c.recommendation.endswith("Post Later") for c in candidates),
            "Needs Human Review":sum("Human Review" in c.recommendation for c in candidates)},
    }
