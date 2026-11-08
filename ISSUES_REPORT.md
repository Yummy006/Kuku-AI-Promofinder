# PromoFinder AI — Issues Report (verified and resolved)

**Review date:** 2026-11-08
**Scope:** Project purpose, correctness, runtime behaviour, tests, and data flow.

## Project motive

PromoFinder AI is an editorial automation tool for finding 20–60 second promotional clips in timestamped screenplay files. It parses episodes, detects hooks and unresolved cliffhangers, scores emotional intensity/context/curiosity, optionally asks an OpenAI-compatible LLM to choose line ranges, validates the choices, ranks them, and exports CSV/JSON.

Its business purpose is to reduce manual clip-selection effort for TikTok, Instagram Reels, and YouTube Shorts. The scores are editorial heuristics, not predictions of views, watch time, or virality; human review remains necessary.

## How this report was verified

Every finding was reproduced (or failed to reproduce) by running the code, not by reading it alone. Windows behaviour was reproduced on Linux by forcing a non-UTF-8 default encoding (`PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 LC_ALL=C`), which gives the same 60 passed / 5 failed result as the original Windows run.

| Check | Before fixes | After fixes |
|---|---|---|
| `python -m compileall -q .` | Pass | Pass |
| `python -m pytest -q` (normal) | 65 passed | **90 passed** |
| `python -m pytest -q` (forced non-UTF-8 default, Windows-like) | 60 passed, 5 failed | **90 passed** |
| `run_all.py sample_data --mock` | Analysed 1 of 3 episodes | Analyses all 3 |
| Non-mock run against a local OpenAI-compatible test server | Worked | Works (correct header, model/URL override, token usage) |
| Wrong API key through the CLI | 14 s, 6 requests per episode, exit code 0 | 1 s, 1 request, exit code **3**, rule-based files still written |
| Streamlit app, all 7 pages with sample data | No exceptions | No exceptions |
| Real LLM provider | Not tested | Not tested (no credential available) |

`tests/test_fixes.py` adds 25 regression tests. Run against the original code, 22 fail; the other 3 pass because they cover behaviour that was already correct (the auth header, retry-then-succeed, and protection of unrelated output files).

## Findings that were retracted

### ISSUE-001 — "LLM authorization header is invalid" — NOT AN ISSUE

The first version of this report said the client sent a masked literal instead of the key. That is wrong for this code: `llm_client.py` builds `Authorization: Bearer <API_KEY>`, and a local test server received `Bearer sk-test-ABC`. A regression test (`test_chat_sends_bearer_header_built_from_the_configured_key`) now locks this in. It passes on the original and the fixed code.

### ISSUE-004 — "CLI ignores Streamlit secrets" — NOT AN ISSUE (as described)

Streamlit exports top-level secrets into environment variables when `llm_client` imports, so the CLI's check already passed with a secrets-only key. The duplication was still real maintenance risk (three copies of the resolution logic), so it was consolidated anyway; see below.

## Findings that were real — all fixed

### ISSUE-002 / ISSUE-003 — Encoding failures on Windows (Medium) — FIXED

The parser's strict UTF-8 reading is correct; the failures came from tests and the report reader using the platform default encoding. All `read_text`/`write_text`/`open` calls in tests, `evaluation/make_report.py` and `run_llm.py` now specify UTF-8. The five failing tests pass on a Windows-like default encoding.

### ISSUE-004 (reframed) — Credential resolution duplicated in three places (Low) — FIXED

New `llm_config.py` is the single resolver (environment `LLM_*`, then `OPENAI_*`, then Streamlit secrets). `config.py` and `llm_client.py` use it, and the CLI checks `llm_client.has_credentials()`.

### ISSUE-005 — Inconsistent deduplication (Low) — FIXED

The analyzer used 50% of the candidate's own duration; ranking used 40% of the shorter clip. New `analysis/overlap.py` provides one `overlap_ratio` / `is_duplicate` (threshold `config.DEDUP_OVERLAP_RATIO`, 40% of the shorter clip), now used by the analyzer, LLM selector, ranking and quality checker. A test asserts the early and global passes agree.

### ISSUE-006 — Parser errors lose their cause (Low) — FIXED

`parse_txt_file_ex()` returns `(episodes, error)` with a readable reason (missing file, permission, undecodable bytes). Encoding is detected (UTF-8, UTF-8 BOM, UTF-16 with BOM) with a documented Windows-1252 fallback that is **reported** (`Episode.encoding`, a terminal warning, and `run_summary.json`). Bytes that decode as none of these are rejected rather than guessed. `parse_txt_file()` keeps its old signature. The Streamlit app uses the same decoder instead of `errors="replace"`, which used to garble non-UTF-8 uploads silently.

## Additional issues found during verification — all fixed

### ISSUE-007 — The app made the CLI refuse to run (Medium) — FIXED

The Streamlit app writes `promo_candidates.json` and `quality_report.json` into `outputs/`; `run_all.py` then stopped with "refusing to clear output directory containing unrelated files". Both names are now managed outputs. Truly unrelated files are still protected (tested).

### ISSUE-008 — A wrong API key wasted time and still exited 0 (Medium) — FIXED

A 401 caused 6 HTTP requests and about 14 seconds per episode (roughly 12 minutes for 50 episodes), then exited 0. Now:

- HTTP 400/401/403/404/422 raise `LLMFatalError` immediately (no retry); the run stops after one request, writes rule-based outputs, and exits 3.
- Only transient errors (network, timeout, 408/429/5xx, malformed reply) are retried.
- Three consecutive unreachable episodes, or every episode failing, also stop the run with exit 3.
- Error messages contain only the HTTP status, never the response body or the key.
- The Streamlit app stops calling the model after a fatal error and falls back to the rule-based engine.

### ISSUE-009 — The sample file held 3 episodes but only 1 was analysed (Low) — FIXED

`sample_screenplay.txt` contains three episodes; the folder runner treated one file as one episode. Files with several `EPISODE NUMBER:` blocks are now split (single-episode files are unaffected). `run_all.py sample_data --mock` reports `Episodes: 3`.

### ISSUE-010 — `run_llm.py` discarded its ranked list (Low) — FIXED

`rank_candidates()` returns a new, de-duplicated list, but `run_llm.py` ignored it, so its outputs kept duplicates and re-added selector flags twice. It now uses the returned list, the shared parser/credential helpers, and fails fast like `run_all.py`.

## Data and privacy considerations

Non-mock runs send episode loglines, continuity notes, and timestamped screenplay lines to the configured LLM endpoint. Operators should process only authorized scripts, verify provider retention/training policies, avoid committing confidential screenplay text, and keep credentials out of logs and reports. API keys are not printed or written to any output file.

## Remaining limitations (not fixed — need real data or a decision)

1. **Model quality is unmeasured.** The LLM path has only been exercised against mocks and a local test server. The README baseline for the rule-based method on the real 50-episode pack (0 Post First, maximum score 61.7) shows the rule-based method alone is weak.
2. Run on the full pack with a real key, add editor-labelled examples, and compare methods (`evaluation/compare.py`, `evaluation/make_report.py`).
3. Cross-episode (multi-segment) promos are not generated; the segment format is ready.
4. The Windows-1252 fallback can mis-decode a rare non-UTF-8 file that happens to be valid cp1252. Saving files as UTF-8 avoids this; a warning is printed whenever the fallback is used.

## Overall assessment

The repository has a coherent purpose and a working offline prototype with clear parser, analysis, scoring, ranking, export, and optional LLM components. With the fixes above the test suite is green on both UTF-8 and Windows-like encodings, the real-provider request path is verified end to end against a test server, and failure modes are fast and visible. It should still be treated as an editorial prototype until it is evaluated on real scripts with a real model and editor-labelled examples.
