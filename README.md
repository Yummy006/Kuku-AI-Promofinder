# PromoFinder AI

## Quick start

```bash
python3 -m pip install -r requirements.txt
export LLM_API_KEY="your-key" LLM_BASE_URL="https://api.openai.com/v1" LLM_MODEL="gpt-4o-mini"
python3 run_all.py "/path/to/show" --limit 3
python3 run_all.py "/path/to/show" --limit 50
streamlit run app.py
```

Hybrid design: Python parses and validates screenplay line IDs, timestamps, duration, scores, tiers and exports; the optional LLM selects candidate line IDs and provides semantic explanations.

## Verify it yourself

```bash
python3 -m pytest -q
python3 run_all.py sample_data --mock
```

`python3 run_all.py sample_data --mock` analyses all three synthetic episodes in the sample file. Mock output is synthetic wiring evidence only; it is not evidence of model quality. Generated `outputs/`, `docs/EVIDENCE.md`, and `evaluation/results.md` are local artifacts and must not be committed or included in a shared source ZIP. Keep only `outputs/.gitkeep` in the repository.

## Measured baseline (user's real 50-episode rule-based run)

- 114 candidates; 112 **Needs Human Review**, 2 **Post Later**, 0 **Post First**.
- Maximum score: 61.7/100.
- 89/114 candidates failed the previous quality checks (21.9% pass rate).
- 98/114 hooks were labelled "Action shock"; roughly 80 hook lines began with camera directions.
- The first two ranked clips were both EP28 and overlapped by approximately 22 seconds.
- One top cliffhanger was a resolved denial ("I didn't!").
- The prior output folder mixed synthetic CSV output with real-pack JSON output.
- The LLM selector had only mock/offline testing at the time of this baseline.

These are supplied baseline measurements, not results from the revised detectors. Re-run both methods before replacing this baseline with measured updated results.

## Input format

The batch runner expects `.txt` files with one episode each (a file that contains several `EPISODE NUMBER:` blocks, like `sample_data/sample_screenplay.txt`, is split into one episode per block). Each file should include `EPISODE NUMBER: Episode N`, an `EPISODE LOGLINE`, and a delimited `--- SCRIPT START ---` / `--- SCRIPT END ---` section. Inside the script section, each scene should have a `SCENE N` heading, `Scene heading`, `Scene start: MM:SS.ss`, and `Scene end: MM:SS.ss`. Dialogue and action/camera lines should include their original start/end timestamps in the parser's documented format. See `parser/txt_parser.py` and `sample_data/sample_screenplay.txt` for a synthetic example.

**Encoding.** Save files as UTF-8 (a UTF-8 BOM and Windows line endings are fine). UTF-16 with a BOM is accepted. As a documented legacy fallback, files that are not valid UTF-8 are decoded as Windows-1252, and the runner prints a warning naming each such file. Files that cannot be decoded, or read at all, are listed with the reason in the terminal and in `run_summary.json` (`input.unparsed_reasons`).

## Run on a new show

```bash
python3 run_all.py "/path/to/new_show_folder" --limit 3 --mock
python3 run_all.py "/path/to/new_show_folder" --limit 50
```

`--mock` is offline wiring-only testing. A normal run requires `LLM_API_KEY` (or `OPENAI_API_KEY`) and sends screenplay text to the configured provider. Each run replaces only the known output files. If `--out` contains unrelated files or directories, the runner stops without deleting them.

Outputs are `rule_based_promos.csv`, `rule_based_promos.json`, `llm_promos.csv`, `llm_promos.json`, `llm_rejected.json`, and `run_summary.json`. The Streamlit app writes `promo_candidates.json` and `quality_report.json` to the same default `outputs/` folder; the runner treats those two as its own, so using the app and then the CLI works.

### When the model cannot be used

| Situation | Behaviour |
|---|---|
| No key (env var or Streamlit secret) | Exit 2 before any work. |
| Wrong key, model name or URL (HTTP 400/401/403/404/422) | The run stops after the first request (these cannot succeed on retry), writes the rule-based outputs, and exits **3**. |
| Network down, timeouts, HTTP 429/5xx | Retried with backoff. After 3 episodes in a row cannot reach the model, the run stops and exits **3**. A few isolated failures only print a warning (details in `llm_rejected.json`). |
| Every episode failed | Exit **3**. |

`run_summary.json` records `llm.aborted_reason`, `llm.episodes_sent_to_llm` and `llm.episodes_failed_to_reach_model`. API keys are never printed or written to any output.

## Swap the model

The standard-library-only client uses an OpenAI-compatible chat-completions endpoint. Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` to change provider/model without editing source. The LLM may choose supplied line IDs only; Python maps IDs to original timestamps and validates the 20–60 second duration. Provider token usage is recorded when returned by the API.


## Segments

Exports include an ordered `segments` list with each segment's episode and start/end timestamps. All current candidates contain one segment. Cross-episode promos are not supported yet; the segment representation is ready for future multi-segment candidates.

## --snap (optional)

Run without `--snap` first and inspect the LLM rejection rate in `run_summary.json`. If it is high (for example, above about 30%) and `rejection_reasons` are mostly hook/cliffhanger position failures, rerun with `--snap` (or set `PROMO_SNAP=1`) and compare the results against your gold labels before keeping it. Snapping trims only to the model-selected hook/cliffhanger line boundaries and rejects clips that fall outside the allowed duration.

## Ranking policy

Candidates are deduplicated within an episode when overlap exceeds 40% of the shorter clip. This one rule (`analysis/overlap.py`, threshold `DEDUP_OVERLAP_RATIO`) is used by the early analysis pass, the LLM selector, global ranking and the quality checker. Severe weaknesses are a hard gate: hook score below 3, cliffhanger below 2, or standalone context below 2 always means **Needs Human Review**, regardless of numeric rank. Among remaining candidates, the first `POST_FIRST_COUNT` eligible clips receive **Post First**, subject to at most two Post First clips per episode. A clip skipped by that episode cap does not consume a Post First slot and may use a **Post Later** slot. The next `POST_LATER_COUNT` eligible clips receive **Post Later**; remaining candidates receive **Needs Human Review**. Numeric scores remain visible.

## Changes

- Review fixes (see `ISSUES_REPORT.md`): explicit UTF-8 everywhere in tests and report reading; encoding-aware parser that reports which encoding it used and why a file failed; one credential/endpoint resolver (`llm_config.py`) used by the CLI, the app and the client; one overlap policy; the CLI accepts the app's output files; wrong credentials fail fast with exit code 3; multi-episode files are split; the app no longer decodes uploads with `errors="replace"`; `run_llm.py` uses the same helpers and keeps its ranked, de-duplicated list.

- Quality-gated ranking now protects severe hook, cliffhanger, and context failures; episode-cap overflow no longer consumes a Post First quota slot.
- Hook, cliffhanger, and context keyword checks use shared word-boundary matching to reduce substring false positives.
- The batch runner reports unparsed files, fails when no episode parses, and supports runtime model/base URL/per-episode overrides.

## Limitations

- The LLM client targets OpenAI-compatible chat-completions endpoints; another API style would need a small adapter because the rest of the code only needs a function `(system, user) -> text`.
- The rule-based hook score counts keyword hits (2.5 per hit), so it cannot separate a single strong event from a weak match; this is a main reason the LLM selector exists.
- Rule-based signals remain heuristics; transfer to a different show's writing style is not established by the supplied single-show baseline.
- The measured baseline comes from one show and had a low validation pass rate.
- The LLM path's quality, latency, and cost need measurement on authorized real scripts and editor-labelled examples.
- The folder CLI parses `.txt` files (each may hold one or several episodes). PDF/DOCX are supported in the Streamlit upload flow, not the batch folder runner.
- Cross-episode promo construction is not supported.
- Scores are editorial heuristics, not predictions of views, watch time, or virality. Human editor review remains necessary.
- Use only screenplay files you are authorized to process. Do not commit real or confidential screenplay text.
