"""Global ranking, overlap deduplication, and rank-based recommendation tiers."""
from typing import List
import pandas as pd
import config
from analysis.screenplay_analyzer import PromoCandidate
from analysis.overlap import overlap_ratio, is_duplicate  # noqa: F401 (overlap_ratio re-exported)

def deduplicate_candidates(candidates: List[PromoCandidate]) -> List[PromoCandidate]:
    """Keep the highest scoring clip when same-episode overlap exceeds configured ratio."""
    ordered=sorted(candidates,key=lambda c:(-c.total_score,c.episode,c.start))
    kept=[]
    for cand in ordered:
        duplicate=next((k for k in kept if is_duplicate(k,cand)),None)
        if duplicate is None:
            kept.append(cand)
    return sorted(kept,key=lambda c:(-c.total_score,c.episode,c.start))

def _has_severe_weakness(candidate: PromoCandidate) -> bool:
    return (candidate.hook_score < 3 or candidate.cliffhanger_score < 2
            or candidate.standalone_score < 2)

def rank_candidates(candidates: List[PromoCandidate]) -> List[PromoCandidate]:
    """Deduplicate, apply quality gates, and fill tier quotas independently.

    Candidates skipped by the Post First per-episode cap do not consume a
    Post First slot; they may still qualify for the Post Later quota.
    """
    ranked = deduplicate_candidates(candidates)
    post_first_total = post_later_total = 0
    post_first_by_episode = {}
    for i, candidate in enumerate(ranked, 1):
        candidate.rank = i
        if _has_severe_weakness(candidate):
            candidate.recommendation = "👁️ Needs Human Review"
            continue
        episode_count = post_first_by_episode.get(candidate.episode, 0)
        if (post_first_total < config.POST_FIRST_COUNT
                and episode_count < config.TOP_LIST_EPISODE_CAP):
            candidate.recommendation = "✅ Post First"
            post_first_total += 1
            post_first_by_episode[candidate.episode] = episode_count + 1
        elif post_later_total < config.POST_LATER_COUNT:
            candidate.recommendation = "🕐 Post Later"
            post_later_total += 1
        else:
            candidate.recommendation = "👁️ Needs Human Review"
    return ranked

def top_n(candidates: List[PromoCandidate], n: int = 10) -> List[PromoCandidate]:
    ranked=rank_candidates(candidates)
    out=[]; per_episode={}
    for c in ranked:
        if c.recommendation != "✅ Post First":
            continue
        if per_episode.get(c.episode,0) >= config.TOP_LIST_EPISODE_CAP:
            continue
        out.append(c); per_episode[c.episode]=per_episode.get(c.episode,0)+1
        if len(out)>=n: break
    return out

def to_dataframe(candidates: List[PromoCandidate]) -> pd.DataFrame:
    rows=[]
    for c in candidates:
        rows.append({
            "Rank":c.rank,"Promo ID":c.promo_id,"Episode":c.episode,
            "Scenes":f"{c.scene_range[0]}–{c.scene_range[1]}",
            "Start":c.start_ts,"End":c.end_ts,"Duration (s)":c.duration,
            "Hook":c.hook_evidence[:80],"Cliffhanger":c.cliffhanger_evidence[:80],
            "Hook Score":c.hook_score,"Cliff Score":c.cliffhanger_score,
            "Standalone":c.standalone_score,"Emotion":c.emotional_intensity,
            "Curiosity":c.viewer_curiosity,"Total Score":c.total_score,
            "Recommendation":c.recommendation,"Why It Works":c.why_it_works})
    return pd.DataFrame(rows)
