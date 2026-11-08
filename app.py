"""PromoFinder AI — complete Streamlit dashboard with seven pages."""
import html
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from parser.txt_parser import (parse_txt_content, parse_directory, decode_screenplay_bytes,
                               split_episode_blocks, ScreenplayReadError)
from parser.pdf_parser import parse_pdf_bytes
from parser.docx_parser import parse_docx_bytes
from analysis.screenplay_analyzer import analyze_all_episodes
from scoring.promo_scorer import score_all
from scoring.ranking import rank_candidates, to_dataframe, top_n
from evaluation.quality_checker import run_quality_check
from export.csv_export import candidates_to_csv_bytes, quality_report_to_csv_bytes
from export.json_export import (
    candidates_to_json_bytes,
    quality_report_to_json_bytes,
    save_outputs,
)
import config

st.set_page_config(
    page_title="PromoFinder AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #0D1117; }
div[data-testid="metric-container"] {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 10px; padding: 14px 20px;
}
div[data-testid="metric-container"] label { color: #8B949E; }
div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: #E63946; font-size: 1.9rem; font-weight: 700;
}
h2 { color: #E63946 !important; }
h3 { color: #F0F6FC !important; }
section[data-testid="stSidebar"] { background: #161B22; }
.promo-card {
    background: #161B22; border: 1px solid #30363D;
    border-left: 4px solid #E63946; border-radius: 8px;
    padding: 18px 22px; margin-bottom: 18px;
}
.promo-card h4 { color: #E63946; margin: 0 0 6px; }
.promo-card p { color: #C9D1D9; margin: 3px 0; font-size: .88rem; }
.badge-post-first { background:#238636; color:white; padding:3px 10px; border-radius:12px; }
.badge-post-later { background:#BB8009; color:white; padding:3px 10px; border-radius:12px; }
.badge-human-review { background:#6E40C9; color:white; padding:3px 10px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "episodes": [],
        "candidates": [],
        "ranked": [],
        "qr_report": {},
        "weights": dict(config.SCORING_WEIGHTS),
        "hr_feedback": {},
        "analysed": False,
        "use_llm": bool(config.USE_LLM),
        "analysis_mode": "Rule-based",
        "llm_rejections": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_pipeline(episodes):
    """Run AI selection when configured; safely fall back to local analysis."""
    llm_rejections = []
    used_llm = False
    with st.spinner("🔍 Analysing screenplays and finding promo moments…"):
        if config.USE_LLM and st.session_state.get("use_llm", False):
            from llm_client import chat, LLMFatalError
            from llm_selector import select_promos
            candidates = []
            llm_available = True
            for episode in episodes:
                if not llm_available:          # key/model/URL is wrong: do not call the model again
                    candidates.extend(analyze_all_episodes([episode]))
                    continue
                try:
                    selected, rejected = select_promos(episode, chat, max_promos=3)
                    llm_rejections.extend(rejected)
                    if selected:
                        candidates.extend(selected)
                        used_llm = True
                    else:
                        # Do not lose a complete episode when the model rejects
                        # every proposal or returns malformed output.
                        candidates.extend(analyze_all_episodes([episode]))
                except LLMFatalError as exc:
                    llm_available = False
                    llm_rejections.append({"episode": episode.number, "reason": str(exc)})
                    st.error(f"LLM selection was stopped: {exc}. Using the rule-based engine instead.")
                    candidates.extend(analyze_all_episodes([episode]))
                except Exception as exc:
                    llm_rejections.append({"episode": episode.number, "reason": str(exc)})
                    candidates.extend(analyze_all_episodes([episode]))
        else:
            candidates = analyze_all_episodes(episodes)

    with st.spinner("📊 Scoring and ranking candidates…"):
        score_all(candidates, st.session_state.weights)
        ranked = rank_candidates(candidates)
    with st.spinner("✅ Checking clip quality and duplicates…"):
        report = run_quality_check(ranked)

    st.session_state.episodes = episodes
    st.session_state.candidates = candidates
    st.session_state.ranked = ranked
    st.session_state.qr_report = report
    st.session_state.analysed = True
    st.session_state.analysis_mode = "LLM-assisted + rule-based fallback" if used_llm else "Rule-based fallback"
    st.session_state.llm_rejections = llm_rejections

    try:
        save_outputs(ranked, report, output_dir=str(PROJECT_DIR / "outputs"))
    except Exception as exc:
        st.warning(f"Could not save output files: {exc}")
    return ranked, report


def safe(value):
    return html.escape(str(value if value is not None else ""))


def promo_card(candidate):
    recommendation = str(candidate.recommendation)
    if "Post First" in recommendation:
        badge = "badge-post-first"
    elif "Post Later" in recommendation:
        badge = "badge-post-later"
    else:
        badge = "badge-human-review"
    score = max(0, min(100, int(candidate.total_score)))
    score_color = "#238636" if score >= 75 else "#BB8009" if score >= 50 else "#E63946"
    card = f"""
    <div class="promo-card">
      <h4>#{safe(candidate.rank)} — {safe(candidate.promo_id)}
        <span class="{badge}">{safe(recommendation)}</span></h4>
      <p><b>Episode:</b> {safe(candidate.episode)} |
         <b>Clip:</b> {safe(candidate.start_ts)} → {safe(candidate.end_ts)}
         ({float(candidate.duration):.1f}s)</p>
      <div style="background:#21262D;border-radius:6px;height:8px;margin:8px 0">
        <div style="background:{score_color};width:{score}%;height:8px;border-radius:6px"></div>
      </div>
      <p><b>Score:</b> {score}/100</p>
      <p><b>Hook:</b> {safe(str(candidate.hook_evidence)[:250])}</p>
      <p><b>Cliffhanger:</b> {safe(str(candidate.cliffhanger_evidence)[:250])}</p>
      <p><b>Context:</b> {safe(getattr(candidate, "standalone_explanation", ""))}</p>
      <p><b>Why it works:</b> {safe(candidate.why_it_works)}</p>
    </div>"""
    st.markdown(card, unsafe_allow_html=True)


init_state()

pages = [
    "🏠 Home Dashboard", "📂 Upload Screenplay", "🔍 AI Analysis",
    "🎞️ Promotional Clips", "⭐ Top Recommendations",
    "✅ Quality Check", "📥 Export Results",
]
with st.sidebar:
    st.markdown("## 🎬 PromoFinder AI")
    st.markdown("*AI-powered Microdrama Promo Discovery*")
    st.divider()
    page = st.radio("Navigate", pages, label_visibility="collapsed")
    st.divider()
    with st.expander("⚙️ Scoring Weights"):
        new_weights = {}
        for key, value in st.session_state.weights.items():
            new_weights[key] = st.slider(
                key.replace("_", " ").title(), 0, 50, int(value),
                key=f"weight_{key}",
            )
        total = sum(new_weights.values())
        st.caption(f"Total: **{total}** (must equal 100)")
        if st.button("Apply Weights", disabled=total != 100):
            st.session_state.weights = new_weights
            if st.session_state.analysed:
                score_all(st.session_state.candidates, new_weights)
                st.session_state.ranked = rank_candidates(st.session_state.candidates)
                st.session_state.qr_report = run_quality_check(st.session_state.ranked)
            st.success("Weights applied.")
    st.caption(f"v{config.APP_VERSION} | Explainable promo discovery")
    if config.USE_LLM:
        st.caption("🟢 LLM API key detected — optional AI selection is available")
    else:
        st.caption("🟡 Offline mode — rule-based analysis is active")


# PAGE 1 — HOME DASHBOARD
if page == "🏠 Home Dashboard":
    st.title("🎬 PromoFinder AI")
    st.subheader("AI-powered Microdrama Promo Discovery System")
    st.write("Find potential 20–60 second promotional moments for social platforms.")
    st.divider()
    ranked = st.session_state.ranked
    episodes = st.session_state.episodes
    report = st.session_state.qr_report
    cols = st.columns(5)
    cols[0].metric("Episodes Loaded", len(episodes))
    cols[1].metric("Scenes Analysed", sum(ep.scene_count for ep in episodes))
    cols[2].metric("Promo Candidates", len(ranked))
    cols[3].metric("Average Score", f"{report.get('average_score', 0):.1f}/100" if report else "—")
    cols[4].metric("Quality Pass Rate", f"{report.get('pass_rate', 0):.0f}%" if report else "—")
    if ranked:
        st.caption(f"Analysis mode: **{st.session_state.analysis_mode}**")
        st.markdown("#### 🥇 Best Ranked Promo")
        promo_card(ranked[0])
        fig = px.histogram(
            x=[c.total_score for c in ranked], nbins=20,
            labels={"x": "Total Score", "y": "Count"},
            color_discrete_sequence=["#E63946"],
        )
        fig.update_layout(plot_bgcolor="#161B22", paper_bgcolor="#161B22", font_color="#F0F6FC")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Go to Upload Screenplay to load files and run analysis.")
        cols = st.columns(4)
        for col, (icon, title, desc) in zip(cols, [
            ("📂", "Upload", "Upload TXT, PDF or DOCX files."),
            ("🔍", "Analyse", "Identify possible hooks and cliffhangers."),
            ("📊", "Score", "Score candidates across five criteria."),
            ("📥", "Export", "Download CSV and JSON results."),
        ]):
            col.markdown(f"### {icon} {title}")
            col.write(desc)


# PAGE 2 — UPLOAD SCREENPLAY
elif page == "📂 Upload Screenplay":
    st.header("📂 Upload Screenplay")
    st.write("Supported formats: TXT, PDF and DOCX.")
    uploaded = st.file_uploader(
        "Drop screenplay files here",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
    st.markdown("**— or —**")
    use_sample = st.button("▶️ Load Sample Screenplay (3 synthetic episodes)")
    if config.USE_LLM:
        st.checkbox("Use LLM to select promo moments (falls back automatically if needed)", key="use_llm")
        st.caption("LLM mode sends screenplay text to the configured provider. Timestamps are still checked by code.")
    else:
        st.info("Optional LLM selection is not configured. The complete local rule-based pipeline works without an API key.")
    with st.expander("Expected screenplay format"):
        st.code("""EPISODE NUMBER: Episode 1
EPISODE LOGLINE: ...
--- SCRIPT START ---
SCENE 01
Scene heading: INT. LOCATION – DAY
Scene start: 00:00.00
Scene end: 01:30.00
**CHARACTER (tone):**
Start: (00:05.00) | End: (00:07.00)
"Dialogue text here."
--- SCRIPT END ---""")

    default_pack = os.getenv("PACK_DIR", "")
    pack_dir = st.text_input("Optional screenplay folder path", value=default_pack,
                             help="Folder containing episode .txt files. You can also set PACK_DIR.")
    if pack_dir and st.button("▶️ Load screenplay folder"):
        if not Path(pack_dir).expanduser().is_dir():
            st.error("Folder not found. Check the path and try again.")
        else:
            with st.spinner("Parsing screenplay folder…"):
                eps = parse_directory(str(Path(pack_dir).expanduser()))
            if eps:
                result, _ = run_pipeline(eps)
                st.success(f"Loaded {len(eps)} episodes; generated {len(result)} candidates.")
            else:
                st.error("No valid episodes were parsed.")

    if uploaded:
        episodes, errors = [], []
        for file in uploaded:
            name = file.name
            ext = name.rsplit(".", 1)[-1].lower()
            data = file.getvalue()
            try:
                parsed = []
                if ext == "txt":
                    text, encoding = decode_screenplay_bytes(data)
                    if encoding not in ("utf-8", "utf-8-sig"):
                        errors.append((name, [f"Decoded as {encoding}, not UTF-8. Check names and punctuation; "
                                              "re-saving the file as UTF-8 is safest."]))
                    for block in split_episode_blocks(text):   # a file may hold several episodes
                        ep = parse_txt_content(block, name)
                        ep.encoding = encoding
                        parsed.append(ep)
                elif ext == "pdf":
                    parsed = [parse_pdf_bytes(data, name)]
                elif ext == "docx":
                    parsed = [parse_docx_bytes(data, name)]
                if not parsed or all(ep is None for ep in parsed):
                    errors.append((name, ["File could not be parsed."]))
                for ep in parsed:
                    if ep is None:
                        continue
                    episodes.append(ep)
                    if getattr(ep, "parse_errors", None):
                        errors.append((name, ep.parse_errors))
            except ScreenplayReadError as exc:
                errors.append((name, [f"Could not read file: {exc}"]))
            except Exception as exc:
                errors.append((name, [str(exc)]))

        if episodes:
            st.success(f"{len(episodes)} episode(s) parsed from {len(uploaded)} file(s).")
            rows = []
            for ep in episodes:
                logline = ep.logline or ""
                rows.append({
                    "Episode": ep.number, "File": ep.filename,
                    "Scenes": ep.scene_count,
                    "Duration": f"{ep.total_duration:.1f}s",
                    "Logline": logline[:70] + ("…" if len(logline) > 70 else ""),
                    "Errors": len(getattr(ep, "parse_errors", [])),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if errors:
            with st.expander("⚠️ Parse warnings"):
                for filename, messages in errors:
                    st.warning(f"{filename}: " + " | ".join(map(str, messages)))
        if episodes and st.button("🚀 Run Analysis"):
            result, _ = run_pipeline(episodes)
            st.success(f"Analysis complete — {len(result)} candidates found.")

    if use_sample:
        sample = PROJECT_DIR / "sample_data" / "sample_screenplay.txt"
        if not sample.exists():
            st.error("sample_data/sample_screenplay.txt not found.")
        else:
            raw = sample.read_text(encoding="utf-8")
            blocks, current = [], []
            for line in raw.splitlines():
                if line.startswith("EPISODE NUMBER:") and current:
                    blocks.append("\n".join(current))
                    current = [line]
                else:
                    current.append(line)
            if current:
                blocks.append("\n".join(current))
            sample_eps = []
            for index, block in enumerate(blocks, start=1):
                if "EPISODE NUMBER:" in block:
                    ep = parse_txt_content(block, f"sample_ep_{index}.txt")
                    if ep:
                        sample_eps.append(ep)
            if sample_eps:
                st.info("This is SYNTHETIC test data, not Kuku's actual show content.")
                result, _ = run_pipeline(sample_eps)
                st.success(f"Loaded {len(sample_eps)} episodes; generated {len(result)} candidates.")
            else:
                st.error("Sample screenplay could not be parsed.")


# PAGE 3 — AI ANALYSIS
elif page == "🔍 AI Analysis":
    st.header("🔍 AI Analysis")
    if not st.session_state.analysed:
        st.info("Upload and analyse a screenplay first (Page 2).")
        st.stop()
    episodes, ranked = st.session_state.episodes, st.session_state.ranked
    if not episodes:
        st.info("No episodes available.")
        st.stop()
    ep_numbers = sorted({ep.number for ep in episodes})
    selected_number = st.selectbox("Select episode to inspect", ep_numbers)
    ep_obj = next((ep for ep in episodes if ep.number == selected_number), None)
    ep_candidates = [c for c in ranked if c.episode == selected_number]
    if ep_obj:
        st.markdown(f"**Logline:** {ep_obj.logline}")
        st.markdown(
            f"**Scenes:** {ep_obj.scene_count} | **Duration:** "
            f"{ep_obj.total_duration:.1f}s | **Promo candidates:** {len(ep_candidates)}"
        )
        st.divider()
        st.subheader("🎭 Scene Breakdown")
        for scene in ep_obj.scenes:
            with st.expander(
                f"Scene {scene.number} — {scene.heading} "
                f"[{scene.start:.1f}s → {scene.end:.1f}s, {scene.duration:.1f}s]"
            ):
                left, right = st.columns(2)
                with left:
                    st.markdown("**Dialogue**")
                    if scene.dialogue:
                        for d in scene.dialogue[:10]:
                            tone_tag = f" *({d.tone})*" if d.tone else ""
                            # Fixed original syntax error: balanced f-string quotes.
                            st.markdown(
                                f'`{d.start:.1f}s` **{d.character}**{tone_tag}: '
                                f'"{d.text[:140]}"'
                            )
                    else:
                        st.caption("No dialogue parsed.")
                with right:
                    st.markdown("**Actions / Camera**")
                    if scene.actions:
                        for action in scene.actions[:8]:
                            st.markdown(f"`{action.start:.1f}s` {action.description[:140]}")
                    else:
                        st.caption("No actions parsed.")
    st.divider()
    st.subheader("📊 Hook & Cliffhanger Scores — Episode Candidates")
    if ep_candidates:
        fig = go.Figure()
        labels = [c.promo_id.split("-")[-1] for c in ep_candidates]
        for name, attr, color in [
            ("Hook", "hook_score", "#E63946"),
            ("Cliffhanger", "cliffhanger_score", "#1F6FEB"),
            ("Standalone", "standalone_score", "#238636"),
            ("Emotion", "emotional_intensity", "#BB8009"),
        ]:
            fig.add_trace(go.Bar(name=name, x=labels, y=[getattr(c, attr) for c in ep_candidates], marker_color=color))
        fig.update_layout(
            barmode="group", plot_bgcolor="#161B22",
            paper_bgcolor="#161B22", font_color="#F0F6FC",
            xaxis_title="Promo ID", yaxis_title="Score (0–10)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No promo candidates found for this episode.")


# PAGE 4 — PROMOTIONAL CLIPS
elif page == "🎞️ Promotional Clips":
    st.header("🎞️ Promotional Clips")
    if not st.session_state.analysed:
        st.info("Upload and analyse a screenplay first (Page 2).")
        st.stop()
    ranked = st.session_state.ranked
    if not ranked:
        st.warning("No promo candidates were generated.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    all_eps = sorted({c.episode for c in ranked})
    selected_eps = c1.multiselect("Episode", all_eps, default=all_eps)
    min_score = c2.slider("Minimum score", 0, 100, 0)
    duration = c3.slider("Duration (seconds)", 10, 65, (20, 60))
    filtered = [
        c for c in ranked
        if c.episode in selected_eps and c.total_score >= min_score
        and duration[0] <= c.duration <= duration[1]
    ]
    st.caption(f"Showing **{len(filtered)}** of {len(ranked)} candidates.")
    if filtered:
        df = to_dataframe(filtered)
        wanted = [
            "Rank", "Promo ID", "Episode", "Start", "End", "Duration (s)",
            "Hook Score", "Cliff Score", "Standalone", "Total Score",
            "Recommendation",
        ]
        st.dataframe(df[[col for col in wanted if col in df.columns]], use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("🕸️ Score Radar — Select a Promo")
        selected_id = st.selectbox("Promo ID", [c.promo_id for c in filtered])
        selected = next(c for c in filtered if c.promo_id == selected_id)
        names = ["Hook", "Cliffhanger", "Standalone", "Emotion", "Curiosity"]
        vals = [
            selected.hook_score, selected.cliffhanger_score,
            selected.standalone_score, selected.emotional_intensity,
            selected.viewer_curiosity,
        ]
        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=names + [names[0]],
            fill="toself", line_color="#E63946",
            fillcolor="rgba(230,57,70,0.2)",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="#161B22",
                radialaxis=dict(visible=True, range=[0, 10], color="#8B949E"),
                angularaxis=dict(color="#8B949E"),
            ),
            paper_bgcolor="#161B22", font_color="#F0F6FC",
            showlegend=False,
        )
        left, right = st.columns(2)
        left.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown(f"**{selected.promo_id}** — EP {selected.episode}")
            st.write(f"Clip: {selected.start_ts} → {selected.end_ts} ({selected.duration:.1f}s)")
            st.write(f"Total score: **{selected.total_score}/100**")
            st.write(f"Recommendation: {selected.recommendation}")
            st.write(f"**Hook:** {selected.hook_evidence}")
            st.write(f"**Cliffhanger:** {selected.cliffhanger_evidence}")
            st.write(f"*{selected.why_it_works}*")
            if getattr(selected, "evidence", None):
                st.text_area("Evidence", str(selected.evidence), height=140, disabled=True)
    else:
        st.info("No candidates match the selected filters.")


# PAGE 5 — TOP RECOMMENDATIONS
elif page == "⭐ Top Recommendations":
    st.header("⭐ Top Recommendations")
    if not st.session_state.analysed:
        st.info("Upload and analyse a screenplay first (Page 2).")
        st.stop()
    ranked = st.session_state.ranked
    if not ranked:
        st.warning("No promo candidates available.")
        st.stop()
    eligible_top = top_n(ranked, len(ranked))
    if not eligible_top:
        st.info("No candidates passed the severe-weakness gate for Post First. Review the full Promotional Clips list and quality flags.")
        st.stop()
    maximum = min(15, len(eligible_top))
    count = st.slider("Show top N Post First promos (maximum 2 per episode)", 1, maximum, min(5, maximum))
    top = eligible_top[:count]
    st.markdown("""
    | Platform | Suggested picks | Ideal duration |
    |---|---:|---:|
    | **TikTok** | Ranks 1–3 | 25–35 s |
    | **Instagram Reels** | Ranks 1–5 | 30–45 s |
    | **YouTube Shorts** | Ranks 1–6 | 40–60 s |
    """)
    st.divider()
    for candidate in top:
        promo_card(candidate)
    st.subheader("📊 Top Promo Score Breakdown")
    labels = [f"#{c.rank} EP{c.episode}" for c in top]
    fig = go.Figure()
    for name, attr, color in [
        ("Hook", "hook_score", "#E63946"),
        ("Cliffhanger", "cliffhanger_score", "#1F6FEB"),
        ("Standalone", "standalone_score", "#238636"),
        ("Emotion", "emotional_intensity", "#BB8009"),
        ("Curiosity", "viewer_curiosity", "#6E40C9"),
    ]:
        fig.add_trace(go.Bar(name=name, x=labels, y=[getattr(c, attr) for c in top], marker_color=color))
    fig.update_layout(
        barmode="group", plot_bgcolor="#161B22",
        paper_bgcolor="#161B22", font_color="#F0F6FC",
        yaxis_title="Score (0–10)",
    )
    st.plotly_chart(fig, use_container_width=True)


# PAGE 6 — QUALITY CHECK
elif page == "✅ Quality Check":
    st.header("✅ Quality Check")
    if not st.session_state.analysed:
        st.info("Upload and analyse a screenplay first (Page 2).")
        st.stop()
    ranked, report = st.session_state.ranked, st.session_state.qr_report
    cols = st.columns(4)
    cols[0].metric("Total Candidates", report.get("total_candidates", 0))
    cols[1].metric("Valid", report.get("valid_candidates", 0))
    cols[2].metric("Flagged", report.get("invalid_candidates", 0))
    cols[3].metric("Pass Rate", f"{report.get('pass_rate', 0):.1f}%")
    errors = report.get("common_errors", [])
    if errors:
        st.subheader("Most Common Issues")
        error_df = pd.DataFrame(errors, columns=["Issue", "Count"])
        fig = px.bar(error_df, x="Count", y="Issue", orientation="h", color_discrete_sequence=["#E63946"])
        fig.update_layout(plot_bgcolor="#161B22", paper_bgcolor="#161B22", font_color="#F0F6FC")
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("🔎 Per-Candidate Review")
    st.caption("Feedback is stored in this Streamlit session.")
    flagged_only = st.checkbox("Show flagged candidates only", value=False)
    display = [c for c in ranked if c.quality_flags] if flagged_only else ranked
    choices = ["—", "👍 Good selection", "👎 Bad selection", "🔧 Needs improvement"]
    for candidate in display:
        flags = candidate.quality_flags or []
        with st.expander(
            f"{'⚠️' if flags else '✅'} {candidate.promo_id} — EP{candidate.episode} "
            f"[{candidate.start_ts}→{candidate.end_ts}] Score: {candidate.total_score}"
        ):
            if flags:
                for flag in flags:
                    st.warning(str(flag))
            else:
                st.success("No quality flags were reported.")
            st.write(f"**Hook evidence:** {candidate.hook_evidence}")
            st.write(f"**Cliffhanger evidence:** {candidate.cliffhanger_evidence}")
            current = st.session_state.hr_feedback.get(candidate.promo_id, "—")
            review = st.radio(
                "Your review", choices, index=choices.index(current),
                key=f"hr_{candidate.promo_id}", horizontal=True,
            )
            if review != "—":
                st.session_state.hr_feedback[candidate.promo_id] = review
                candidate.human_review = review


# PAGE 7 — EXPORT RESULTS
elif page == "📥 Export Results":
    st.header("📥 Export Results")
    if not st.session_state.analysed:
        st.info("Upload and analyse a screenplay first (Page 2).")
        st.stop()
    ranked, report = st.session_state.ranked, st.session_state.qr_report
    for candidate in ranked:
        if candidate.promo_id in st.session_state.hr_feedback:
            candidate.human_review = st.session_state.hr_feedback[candidate.promo_id]

    st.subheader("📊 Analysis Results")
    st.caption(f"Analysis mode: {st.session_state.analysis_mode}")
    approved = [c for c in ranked if "Good selection" in (c.human_review or "")]
    st.metric("Human-approved promos", len(approved))
    left, right = st.columns(2)
    left.download_button(
        "⬇️ Download Promo Candidates (CSV)",
        data=candidates_to_csv_bytes(ranked),
        file_name="promo_candidates.csv", mime="text/csv",
    )
    right.download_button(
        "⬇️ Download Promo Candidates (JSON)",
        data=candidates_to_json_bytes(ranked),
        file_name="promo_candidates.json", mime="application/json",
    )
    st.download_button(
        "⬇️ Download Human-approved Promos (CSV)",
        data=candidates_to_csv_bytes(approved),
        file_name="approved_promos.csv", mime="text/csv",
        disabled=not approved,
        help="Mark clips as Good selection in Quality Check to include them here.",
    )
    if st.session_state.llm_rejections:
        with st.expander("LLM proposal validation / fallback log"):
            st.json(st.session_state.llm_rejections)
    st.divider()
    st.subheader("✅ Quality Report")
    left, right = st.columns(2)
    left.download_button(
        "⬇️ Download Quality Report (CSV)",
        data=quality_report_to_csv_bytes(report, ranked),
        file_name="quality_report.csv", mime="text/csv",
    )
    right.download_button(
        "⬇️ Download Quality Report (JSON)",
        data=quality_report_to_json_bytes(report, ranked),
        file_name="quality_report.json", mime="application/json",
    )

    summary = {
        "total_episodes": len(st.session_state.episodes),
        "total_scenes": sum(ep.scene_count for ep in st.session_state.episodes),
        "total_candidates": len(ranked),
        "average_score": report.get("average_score", 0),
        "pass_rate_pct": report.get("pass_rate", 0),
        "top_5_promos": [
            {
                "rank": c.rank, "promo_id": c.promo_id, "episode": c.episode,
                "start": c.start_ts, "end": c.end_ts, "score": c.total_score,
                "recommendation": c.recommendation,
            }
            for c in ranked[:5]
        ],
        "scoring_weights_used": st.session_state.weights,
        "analysis_mode": st.session_state.analysis_mode,
        "human_reviews": st.session_state.hr_feedback,
        "llm_rejections": st.session_state.llm_rejections,
    }
    st.divider()
    st.subheader("📄 Analysis Summary")
    left, right = st.columns([1, 2])
    left.download_button(
        "⬇️ Download Analysis Summary (JSON)",
        data=json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="analysis_summary.json", mime="application/json",
    )
    with right:
        with st.expander("📋 Preview summary"):
            st.json(summary)

    st.divider()
    st.subheader("📁 Output Files (saved on disk)")
    output_dir = PROJECT_DIR / "outputs"
    if output_dir.is_dir():
        files = sorted(path for path in output_dir.iterdir() if path.is_file())
        if files:
            for path in files:
                st.write(f"📄 outputs/{path.name} — {path.stat().st_size:,} bytes")
        else:
            st.caption("No output files yet.")
    else:
        st.caption("The outputs/ folder has not been created yet.")
