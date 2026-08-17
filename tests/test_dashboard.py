"""Invariant tests for the free-LLM availability dashboard.

These tests run offline (no live probes, no network) and enforce the
policy invariants that keep the repo honest:

- no literal provider API keys anywhere in source
- .env.example documents every key the dashboard reads
- probe/generate scripts behave (pure functions, no surprises)
"""

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO / ".env.example"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return load_module("probe_models", "probe_models.py")


@pytest.fixture(scope="module")
def common():
    return load_module("common", "common.py")


# ── Security: no literal provider keys ────────────────────────────────────────


PROVIDER_KEY_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{16,}"),  # OpenRouter
    re.compile(r"gsk_[A-Za-z0-9_-]{16,}"),  # Groq
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),  # Google AI Studio
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub
    re.compile(r"hf_[A-Za-z0-9]{20,}"),  # HuggingFace
    re.compile(r"nvapi-[A-Za-z0-9_-]{16,}"),  # NVIDIA NIM
    re.compile(r"sk-lf-[A-Za-z0-9_-]{16,}"),  # Langfuse
]

SKIP_DIRS = {".git", "docs", ".venv", "__pycache__", "node_modules"}
# .env / .env.local hold real local keys (gitignored); .env.example is the
# placeholder template and probes.jsonl is generated probe data.
SKIP_FILES = {".env", ".env.local", ".env.example", "probes.jsonl"}
BINARY_EXTS = {".jsonl", ".gz", ".png", ".jpg", ".svg", ".ico", ".woff2", ".db", ".sqlite3"}


def iter_source_files():
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.name in SKIP_FILES:
            continue
        if p.suffix.lower() in BINARY_EXTS:
            continue
        yield p


def test_no_literal_provider_keys_committed():
    """Source files must not contain literal provider API keys."""
    offenders = []
    for p in iter_source_files():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat in PROVIDER_KEY_PATTERNS:
                m = pat.search(line)
                if m:
                    # Allow obvious placeholders like KEY= or example.
                    offenders.append(f"{p.relative_to(REPO)}:{i}: {pat.pattern[:20]}...")
                    break
    assert not offenders, "Literal provider keys found:\n" + "\n".join(offenders)


def test_env_example_covers_dashboard_env_vars():
    """Every os.environ key read by the dashboard scripts must be documented."""
    template = ENV_EXAMPLE.read_text()
    env_keys = set()
    for name, path in [("probe_models", "probe_models.py"), ("generate_site", "generate_site.py")]:
        try:
            load_module(name, path)
        except Exception:
            continue
        src = (REPO / path).read_text()
        env_keys |= set(re.findall(r'os\.environ\.get\("([A-Z0-9_]+)"', src))
    for key in sorted(env_keys):
        assert re.search(rf"^{re.escape(key)}=", template, re.M), (
            f"os.environ key {key} missing from .env.example"
        )


# ── common.py ─────────────────────────────────────────────────────────────────


def test_is_free_floats(common):
    assert common._is_free({"price": 0}, "price") is True
    assert common._is_free({"price": 0.0}, "price") is True
    assert common._is_free({"price": 0.5}, "price") is False


def test_is_free_missing_or_garbage(common):
    assert common._is_free({}, "price") is False
    assert common._is_free({"price": None}, "price") is False
    assert common._is_free({"price": "not-a-number"}, "price") is False


# ── probe_models.py pure functions ────────────────────────────────────────────


def test_classify_ok(probe):
    body = json.dumps({"choices": [{"text": "hi"}]})
    assert probe.classify(200, body, None) == "ok"
    body = json.dumps({"candidates": [{"content": "hi"}]})
    assert probe.classify(200, body, None) == "ok"


def test_classify_http_errors(probe):
    assert probe.classify(429, "", None) == "rate_limited"
    assert probe.classify(401, "", None) == "auth_error"
    assert probe.classify(403, "", None) == "auth_error"
    assert probe.classify(402, "", None) == "quota_exhausted"
    assert probe.classify(404, "", None) == "not_found"
    assert probe.classify(502, "", None) == "server_error"
    assert probe.classify(400, "", None) == "bad_response"


def test_classify_exception_and_timeout(probe):
    assert probe.classify(None, "", None) == "network_error"
    assert probe.classify(None, "", TimeoutError("timed out")) == "timeout"
    assert probe.classify(None, "", OSError("boom")) == "network_error"


def test_classify_200_with_error_field(probe):
    assert probe.classify(200, json.dumps({"error": {"message": "model_not_found"}}), None) == "not_found"
    assert probe.classify(200, json.dumps({"error": "rate limit exceeded"}), None) == "rate_limited"


def test_bucket_and_run_index(probe):
    from datetime import datetime

    # Deterministic: same model → same bucket.
    assert probe.bucket_for("gpt-5") == probe.bucket_for("gpt-5")
    # With a single bucket everything is in bucket 0.
    assert 0 <= probe.bucket_for("anything") < probe.ROUND_ROBIN_BUCKETS
    assert 0 <= probe.run_index_for(datetime.now()) < probe.ROUND_ROBIN_BUCKETS


def test_percentile(probe):
    assert probe.percentile([], 50) is None
    assert probe.percentile([1, 2, 3], 50) == 2
    assert probe.percentile([1, 2, 3], 95) == 3
    assert probe.percentile([5], 50) == 5


def test_is_on_watch_list(probe):
    assert probe.is_on_watch_list(["ok", "ok"]) is False
    assert probe.is_on_watch_list(["fail", "fail", "fail"]) is True
    assert probe.is_on_watch_list(["fail", "fail"]) is False  # not enough fails


def test_load_recent_statuses_missing_file(probe):
    assert probe.load_recent_statuses(Path(tempfile.gettempdir()) / "nope.jsonl") == {}


def test_aggregate_and_rotate_empty(probe, tmp_path):
    assert probe.aggregate_and_rotate(tmp_path / "missing.jsonl") == {}


def test_aggregate_and_rotate_basic(probe, tmp_path):
    from datetime import datetime, timedelta, timezone

    probes = tmp_path / "probes.jsonl"
    now = datetime.now(timezone.utc)
    lines = []
    # 4 ok probes today, 1 rate_limited today
    for i in range(4):
        lines.append(
            json.dumps(
                {
                    "ts": now.isoformat(),
                    "provider": "groq",
                    "model": "llama-3.3-70b",
                    "status": "ok",
                    "latency_ms": 100 + i,
                }
            )
        )
    lines.append(
        json.dumps(
            {
                "ts": now.isoformat(),
                "provider": "groq",
                "model": "llama-3.3-70b",
                "status": "rate_limited",
                "latency_ms": 500,
            }
        )
    )
    # 1 old probe (>30d) → archived
    lines.append(
        json.dumps(
            {
                "ts": (now - timedelta(days=40)).isoformat(),
                "provider": "groq",
                "model": "old-model",
                "status": "ok",
            }
        )
    )
    probes.write_text("".join(line + "\n" for line in lines))

    out = probe.aggregate_and_rotate(probes, keep_days=30)
    groq = out["groq"]["llama-3.3-70b"]
    assert groq["samples_7d"] == 5
    assert groq["uptime_7d"] == pytest.approx(0.8)
    assert groq["rate_limited_7d"] == 1
    assert groq["down_reason_7d"] == "rate_limited"
    assert groq["p50_latency_ms"] == 102  # percentile([100,101,102,103], 50) → round(1.5)=2 → 102
    assert "old-model" not in out["groq"]
    # The old line should have been rotated out of the live file.
    remaining = probes.read_text().splitlines()
    assert all("old-model" not in line for line in remaining)


def test_build_request_openai_style(probe, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    cfg = probe.PROVIDER_PROBES["groq"]
    url, headers, body = probe.build_request("groq", cfg, "llama-3.3-70b", "deadbeef")
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    payload = json.loads(body)
    assert payload["model"] == "llama-3.3-70b"
    assert payload["max_tokens"] == 1
    assert payload["messages"][0]["content"] == "ping deadbeef"


def test_build_request_gemini_style(probe, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = probe.PROVIDER_PROBES["gemini"]
    url, headers, body = probe.build_request("gemini", cfg, "models/gemini-2.0-flash", "aabb")
    assert "generateContent" in url
    assert url.endswith("/models/gemini-2.0-flash:generateContent")
    assert headers["x-goog-api-key"] == "test-key"
    payload = json.loads(body)
    assert "contents" in payload


def test_build_request_cloudflare_requires_account(probe, monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    cfg = probe.PROVIDER_PROBES["cloudflare"]
    assert probe.build_request("cloudflare", cfg, "some-model", "cccc") is None


def test_probe_help_runs():
    result = subprocess.run(
        [sys.executable, str(REPO / "probe_models.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


# ── .env.example shape ────────────────────────────────────────────────────────


def test_env_example_has_provider_keys():
    """The dashboard template must document at least the major provider keys."""
    text = ENV_EXAMPLE.read_text()
    for key in [
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "HF_TOKEN",
        "NVIDIA_NIM_API_KEY",
        "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_ACCOUNT_ID",
    ]:
        assert re.search(rf"^{re.escape(key)}=", text, re.M), f"{key} missing from .env.example"


def test_env_example_has_no_proxy_remnants():
    """Proxy-era variables must not linger in the dashboard template."""
    text = ENV_EXAMPLE.read_text()
    for key in ["LITELLM_MASTER_KEY", "DATABASE_URL", "POSTGRES_PASSWORD", "SYNC_INTERVAL_HOURS"]:
        assert not re.search(rf"^{re.escape(key)}=", text, re.M), (
            f"proxy-era {key} still in .env.example"
        )
