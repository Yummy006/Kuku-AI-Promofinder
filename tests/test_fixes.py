"""Regression tests for the issues found in the review (see ISSUES_REPORT.md).

All file I/O uses explicit UTF-8, so these pass on Windows as well as Linux/macOS.
No real network calls are made: HTTP is either mocked or served from localhost.
"""
import io
import json
import re
import subprocess
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT / "sample_data" / "sample_screenplay.txt"


def _sample_blocks():
    raw = SAMPLE.read_text(encoding="utf-8")
    return [b for b in re.split(r"(?=EPISODE NUMBER:)", raw) if "EPISODE NUMBER:" in b]


def _candidate(episode, start, end, pid="X"):
    from analysis.screenplay_analyzer import PromoCandidate
    return PromoCandidate(promo_id=pid, episode=episode, scene_range=(1, 1), start=start, end=end,
                          duration=end - start, start_ts="", end_ts="")


# ----------------------------------------------------------------- ISSUE-001 (header)
class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok_response():
    body = {"choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
    return _FakeResponse(json.dumps(body).encode("utf-8"))


def test_chat_sends_bearer_header_built_from_the_configured_key(monkeypatch):
    import llm_client
    monkeypatch.setattr(llm_client, "API_KEY", "sk-unit-test-key")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["url"] = req.full_url
        return _ok_response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert llm_client.chat("system", "user") == "ok"
    assert seen["headers"]["authorization"] == "Bearer sk-unit-test-key"
    assert seen["url"].endswith("/chat/completions")


def test_error_messages_never_contain_the_api_key(monkeypatch):
    import llm_client
    monkeypatch.setattr(llm_client, "API_KEY", "sk-SECRET-VALUE")
    err = urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, io.BytesIO(b'{"key":"sk-SECRET-VALUE"}'))
    monkeypatch.setattr("urllib.request.urlopen", mock.Mock(side_effect=err))
    with pytest.raises(llm_client.LLMFatalError) as info:
        llm_client.chat("s", "u")
    assert "sk-SECRET-VALUE" not in str(info.value)
    assert "401" in str(info.value)


# ------------------------------------------------- bad key: fail fast, retry only transient
def test_auth_error_is_fatal_and_not_retried(monkeypatch):
    import llm_client
    monkeypatch.setattr(llm_client, "API_KEY", "k")
    for code in (400, 401, 403, 404):
        err = urllib.error.HTTPError("http://x", code, "Nope", {}, io.BytesIO(b""))
        opener = mock.Mock(side_effect=err)
        monkeypatch.setattr("urllib.request.urlopen", opener)
        with pytest.raises(llm_client.LLMFatalError):
            llm_client.chat("s", "u")
        assert opener.call_count == 1, f"HTTP {code} must not be retried"


def test_transient_errors_are_retried_then_raise_llmerror(monkeypatch):
    import llm_client
    monkeypatch.setattr(llm_client, "API_KEY", "k")
    monkeypatch.setattr("time.sleep", lambda s: None)
    for code in (429, 500, 503):
        err = urllib.error.HTTPError("http://x", code, "Busy", {}, io.BytesIO(b""))
        opener = mock.Mock(side_effect=err)
        monkeypatch.setattr("urllib.request.urlopen", opener)
        with pytest.raises(llm_client.LLMError) as info:
            llm_client.chat("s", "u", retries=3)
        assert not isinstance(info.value, llm_client.LLMFatalError)
        assert opener.call_count == 3


def test_transient_error_then_success_recovers(monkeypatch):
    import llm_client
    monkeypatch.setattr(llm_client, "API_KEY", "k")
    monkeypatch.setattr("time.sleep", lambda s: None)
    err = urllib.error.HTTPError("http://x", 429, "Slow down", {}, io.BytesIO(b""))
    monkeypatch.setattr("urllib.request.urlopen", mock.Mock(side_effect=[err, _ok_response()]))
    assert llm_client.chat("s", "u") == "ok"


def test_select_promos_propagates_fatal_errors_and_marks_transport_errors():
    import llm_client
    from llm_selector import select_promos
    from parser.txt_parser import parse_txt_content
    ep = parse_txt_content(_sample_blocks()[0], "e1.txt")

    def fatal(_s, _u):
        raise llm_client.LLMFatalError("HTTP 401")

    with pytest.raises(llm_client.LLMFatalError):
        select_promos(ep, fatal)

    calls = []

    def transient(_s, _u):
        calls.append(1)
        raise llm_client.LLMError("timeout")

    got, rejected = select_promos(ep, transient)
    assert got == [] and len(calls) == 1          # no second round of waiting
    assert rejected[0]["transport_error"] is True


# ------------------------------------------------------------- ISSUE-004 (credentials)
def test_credentials_resolve_in_one_place(monkeypatch):
    import llm_config
    for name in ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_config, "_from_secrets", lambda *names: "")
    assert llm_config.resolve_llm_settings().api_key == ""
    monkeypatch.setenv("OPENAI_API_KEY", "fallback")
    assert llm_config.resolve_llm_settings().api_key == "fallback"
    monkeypatch.setenv("LLM_API_KEY", "primary")
    assert llm_config.resolve_llm_settings().api_key == "primary"
    monkeypatch.setenv("LLM_MODEL", "my-model")
    assert llm_config.resolve_llm_settings().model == "my-model"


def test_secrets_are_used_when_env_is_empty(monkeypatch):
    import llm_config
    for name in ("LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_config, "_from_secrets",
                        lambda *names: "from-secrets" if "LLM_API_KEY" in names else "")
    assert llm_config.resolve_llm_settings().api_key == "from-secrets"


def test_config_and_client_share_the_resolver():
    import config, llm_client
    assert config.USE_LLM == llm_client.has_credentials()


# ------------------------------------------------------------- ISSUE-005 (dedupe policy)
def test_one_overlap_policy_everywhere():
    import config
    from analysis.overlap import overlap_ratio, is_duplicate
    from analysis.screenplay_analyzer import _deduplicate
    from scoring.ranking import deduplicate_candidates
    short = _candidate(1, 0, 20, "short")
    long_ = _candidate(1, 10, 60, "long")          # overlap 10s = 50% of the shorter clip, 20% of the longer
    assert overlap_ratio(short, long_) == pytest.approx(0.5)
    assert config.DEDUP_OVERLAP_RATIO == 0.40
    assert is_duplicate(short, long_)
    # The early (analysis) pass and the global (ranking) pass now agree.
    assert len(_deduplicate([short, long_])) == 1
    assert len(deduplicate_candidates([_candidate(1, 0, 20, "a"), _candidate(1, 10, 60, "b")])) == 1
    # 30% overlap of the shorter clip is below the 40% threshold: both pass, in both places.
    a, b = _candidate(1, 0, 20, "a"), _candidate(1, 14, 64, "b")
    assert len(_deduplicate([a, b])) == 2
    assert len(deduplicate_candidates([_candidate(1, 0, 20, "a"), _candidate(1, 14, 64, "b")])) == 2


def test_clips_from_different_episodes_are_never_duplicates():
    from analysis.overlap import is_duplicate
    assert not is_duplicate(_candidate(1, 0, 30), _candidate(2, 0, 30))


# ----------------------------------------------- ISSUE-002 / 006 (encoding, error causes)
def test_decode_reports_the_encoding_used():
    from parser.txt_parser import decode_screenplay_bytes
    text = "Maya — “Richard?” café"
    assert decode_screenplay_bytes(text.encode("utf-8")) == (text, "utf-8")
    assert decode_screenplay_bytes(b"\xef\xbb\xbf" + text.encode("utf-8")) == (text, "utf-8-sig")
    assert decode_screenplay_bytes(text.encode("utf-16")) == (text, "utf-16")
    assert decode_screenplay_bytes(text.encode("cp1252")) == (text, "cp1252")


def test_undecodable_bytes_raise_instead_of_being_accepted():
    from parser.txt_parser import decode_screenplay_bytes, ScreenplayReadError
    with pytest.raises(ScreenplayReadError):
        decode_screenplay_bytes(b"\x81\x8d\x8f\x90 not text")


def test_parse_txt_file_ex_returns_the_cause(tmp_path):
    from parser.txt_parser import parse_txt_file_ex, parse_txt_file
    eps, err = parse_txt_file_ex(str(tmp_path / "missing.txt"))
    assert eps == [] and "FileNotFoundError" in err
    assert parse_txt_file(str(tmp_path / "missing.txt")) is None     # old API still works
    eps, err = parse_txt_file_ex(str(tmp_path))                        # a directory
    assert eps == [] and "cannot read file" in err
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\x81\x8d\x8f\x90 not text")
    eps, err = parse_txt_file_ex(str(bad))
    assert eps == [] and "re-save" in err


def test_legacy_windows_file_is_parsed_and_the_encoding_is_reported(tmp_path):
    from parser.txt_parser import parse_txt_file_ex
    block = _sample_blocks()[0].replace("Maya", "Maya").replace("\n", "\n", 1)
    legacy = tmp_path / "episode_1.txt"
    legacy.write_bytes(block.replace("’", "'").replace("“", '"').replace("”", '"').encode("cp1252", errors="replace"))
    eps, err = parse_txt_file_ex(str(legacy))
    assert err is None and len(eps) == 1 and eps[0].scene_count > 0
    assert eps[0].encoding in ("utf-8", "cp1252")


def test_bom_and_crlf_files_parse(tmp_path):
    from parser.txt_parser import parse_txt_file_ex
    block = _sample_blocks()[0]
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbf" + block.encode("utf-8"))
    (tmp_path / "crlf.txt").write_bytes(block.replace("\n", "\r\n").encode("utf-8"))
    for name in ("bom.txt", "crlf.txt"):
        eps, err = parse_txt_file_ex(str(tmp_path / name))
        assert err is None and eps[0].scene_count > 0 and eps[0].number == 1


# ------------------------------------------------------- multi-episode sample file
def test_one_file_with_three_episodes_yields_three_episodes():
    from parser.txt_parser import parse_txt_file_ex, split_episode_blocks
    eps, err = parse_txt_file_ex(str(SAMPLE))
    assert err is None
    assert [e.number for e in eps] == [1, 2, 3]
    assert all(e.scene_count > 0 for e in eps)
    assert len(split_episode_blocks("no marker here")) == 1
    assert len(split_episode_blocks(_sample_blocks()[0])) == 1


def test_run_all_on_sample_data_analyses_all_three_episodes(tmp_path):
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, "run_all.py", "sample_data", "--mock", "--out", str(out)],
                            cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Episodes: 3" in result.stdout
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["episodes"] == 3 and summary["input"]["episodes_parsed"] == 3


def test_run_all_reports_why_a_file_was_unreadable(tmp_path):
    folder = tmp_path / "show"
    folder.mkdir()
    (folder / "episode_1.txt").write_text(_sample_blocks()[0], encoding="utf-8")
    (folder / "garbage.txt").write_bytes(b"\x81\x8d\x8f\x90 not text")
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, "run_all.py", str(folder), "--mock", "--out", str(out)],
                            cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "garbage.txt:" in result.stderr and "re-save" in result.stderr
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert "re-save" in summary["input"]["unparsed_reasons"]["garbage.txt"]


def test_run_all_warns_when_a_file_needed_the_fallback_encoding(tmp_path):
    folder = tmp_path / "show"
    folder.mkdir()
    block = _sample_blocks()[0] + "\nNOTE: café\n"
    (folder / "episode_1.txt").write_bytes(block.encode("cp1252"))
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, "run_all.py", str(folder), "--mock", "--out", str(out)],
                            cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "cp1252" in result.stderr
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["input"]["non_utf8_files"] == {"episode_1.txt": "cp1252"}


# --------------------------------------- app output files must not block the CLI
def test_cli_accepts_an_outputs_folder_the_app_already_wrote_to(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "promo_candidates.json").write_text("[]", encoding="utf-8")
    (out / "quality_report.json").write_text("{}", encoding="utf-8")
    result = subprocess.run([sys.executable, "run_all.py", "sample_data", "--mock", "--out", str(out)],
                            cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_still_protects_unrelated_files(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "my_notes.txt").write_text("keep", encoding="utf-8")
    result = subprocess.run([sys.executable, "run_all.py", "sample_data", "--mock", "--out", str(out)],
                            cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode != 0 and (out / "my_notes.txt").read_text(encoding="utf-8") == "keep"


# --------------------------------- real CLI against a local server: bad key aborts fast
class _Handler(BaseHTTPRequestHandler):
    status = 401
    hits = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).hits.append(self.headers.get("Authorization"))
        data = json.dumps({"error": {"message": "nope"}}).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve(status):
    handler = type("H", (_Handler,), {"status": status, "hits": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler


def test_wrong_key_stops_the_run_after_one_request_with_nonzero_exit(tmp_path):
    server, handler = _serve(401)
    try:
        import os
        env = dict(os.environ, LLM_API_KEY="sk-wrong",
                   LLM_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1")
        out = tmp_path / "out"
        result = subprocess.run([sys.executable, "run_all.py", "sample_data", "--out", str(out)],
                                cwd=PROJECT, env=env, capture_output=True, text=True, timeout=60)
    finally:
        server.shutdown()
    assert result.returncode == 3
    assert len(handler.hits) == 1                  # 3 episodes, but only ONE request was made
    assert handler.hits[0] == "Bearer sk-wrong"
    assert "aborted" in result.stderr and "sk-wrong" not in result.stdout + result.stderr
    assert (out / "rule_based_promos.csv").exists()   # rule-based results are still delivered
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["llm"]["aborted_reason"] and summary["llm"]["episodes_sent_to_llm"] == 1


def _run_main(monkeypatch, argv, chat):
    import run_all, llm_client
    monkeypatch.setattr(llm_client, "API_KEY", "k")
    monkeypatch.setattr(llm_client, "chat", chat)
    monkeypatch.setattr(sys, "argv", ["run_all.py"] + argv)
    return run_all.main()


def _five_episode_folder(tmp_path):
    folder = tmp_path / "show"
    folder.mkdir()
    for n in range(1, 6):
        block = re.sub(r"EPISODE NUMBER:\s*Episode\s*\d+", f"EPISODE NUMBER: Episode {n}", _sample_blocks()[0], count=1)
        (folder / f"episode_{n:02}.txt").write_text(block, encoding="utf-8")
    return folder


def test_unreachable_model_aborts_after_three_consecutive_failures(tmp_path, monkeypatch):
    import llm_client
    calls = []

    def unreachable(_s, _u):
        calls.append(1)
        raise llm_client.LLMError("connection refused")

    code = _run_main(monkeypatch, [str(_five_episode_folder(tmp_path)), "--out", str(tmp_path / "o")], unreachable)
    assert code == 3
    assert len(calls) == 3                          # 5 episodes, stopped after 3 in a row


def test_a_good_episode_resets_the_failure_streak(tmp_path, monkeypatch):
    import llm_client, run_all
    state = {"n": 0}

    def flaky(system, user):
        state["n"] += 1
        if state["n"] in (1, 2, 4, 5):             # fail, fail, ok, fail, fail -> never 3 in a row
            raise llm_client.LLMError("timeout")
        return run_all.mock_client(system, user)

    code = _run_main(monkeypatch, [str(_five_episode_folder(tmp_path)), "--out", str(tmp_path / "o")], flaky)
    assert state["n"] == 5 and code == 0            # all episodes attempted; partial failure is a warning
