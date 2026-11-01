"""Tests for the relay→dashboard latency feedback pipeline.

docs/relay-latency.json carries MEASURED relay traffic p50s
(scripts/relay_latency_feedback.py), which generate_site.select_dynamic_chain
uses to order same-model provider groups. 1-token probes cannot see the
production gap (2026-11-01: openrouter Nemotron 550B p50 545ms vs NVIDIA
3670ms while probes read 666ms vs 778ms).

Invariants under test:
- both vercel log shapes parse (new {"type":"log","data":{}} wrap + legacy)
- only provider_success events count; junk lines are skipped
- p50/p95 are nearest-rank over the sorted latency list
- a zero-success run never wipes an existing non-empty relay-latency.json
- fresh data with n >= 3 orders a group by p50; stale/thin/missing data
  falls back to RELAY_PROVIDER_PREFERENCE / availability order
"""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "relay_latency_feedback.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def feedback():
    return load_module("relay_latency_feedback", SCRIPT)


@pytest.fixture(scope="module")
def gs():
    return load_module("generate_site", "generate_site.py")


TS = "2026-11-01T06:32:00+00:00"


def _message(event="provider_success", provider="nvidia-nemotron3ultra550ba55b",
             model="nvidia/nemotron-3-ultra-550b-a55b", latency=3670):
    return (
        f"[relay] req=abc event={event} provider={provider} "
        f"model={model} status=200 retry=0 latency={latency}ms"
    )


def _new_wrap(**kwargs):
    return json.dumps({"type": "log", "data": {"message": _message(**kwargs), "timestamp": TS}})


def _legacy(**kwargs):
    return json.dumps({"message": _message(**kwargs), "timestamp": TS})


# ── parsing ───────────────────────────────────────────────────────────────────


def test_parse_new_cli_wrap(feedback):
    rec = feedback.parse_log_line(_new_wrap())
    assert rec is not None
    assert rec["event"] == "provider_success"
    assert rec["provider"] == "nvidia-nemotron3ultra550ba55b"
    assert rec["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert rec["latency_ms"] == 3670
    assert rec["timestamp"] == datetime(2026, 11, 1, 6, 32, tzinfo=timezone.utc)


def test_parse_legacy_top_level(feedback):
    rec = feedback.parse_log_line(
        _legacy(provider="openrouter-nemotron3ultra550ba55b",
                model="nvidia/nemotron-3-ultra-550b-a55b:free", latency=545)
    )
    assert rec is not None
    assert rec["provider"] == "openrouter-nemotron3ultra550ba55b"
    assert rec["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert rec["latency_ms"] == 545


def test_parse_skips_non_success_and_junk(feedback):
    assert feedback.parse_log_line("") is None
    assert feedback.parse_log_line("not json at all") is None
    assert feedback.parse_log_line(json.dumps([1, 2, 3])) is None
    assert feedback.parse_log_line(_new_wrap(event="provider_error")) is None
    # Success event missing a required field is unusable.
    line = json.dumps({"data": {"message": "[relay] event=provider_success provider=zen"}})
    assert feedback.parse_log_line(line) is None


# ── aggregation math ─────────────────────────────────────────────────────────


def test_percentiles_nearest_rank(feedback):
    assert feedback.percentile([], 50) == 0
    assert feedback.percentile([42], 50) == 42
    assert feedback.percentile([10, 20, 30, 40], 50) == 20
    assert feedback.percentile([10, 20, 30, 40], 95) == 40


def test_build_report_aggregates_per_provider_model(feedback):
    records = [
        feedback.parse_log_line(json.dumps({"message": _message(latency=v), "timestamp": TS}))
        for v in (100, 200, 300, 400, 1000)
    ]
    now = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
    report = feedback.build_report(records, now=now)
    rec = report["providers"]["nvidia-nemotron3ultra550ba55b"][
        "nvidia/nemotron-3-ultra-550b-a55b"
    ]
    assert rec["n"] == 5
    assert rec["p50_ms"] == 300  # nearest-rank index 2
    assert rec["p95_ms"] == 1000  # nearest-rank last
    assert rec["last"] == TS
    assert report["window"] == {"first": TS, "last": TS}
    assert report["generated_at_utc"] == now.isoformat()


# ── regression guard: never empty an existing file ───────────────────────────


def _write(path, providers):
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": TS,
                "window": {"first": TS, "last": TS},
                "providers": providers,
            }
        ),
        encoding="utf-8",
    )


def test_empty_input_keeps_existing_file(feedback, tmp_path, capsys):
    out = tmp_path / "relay-latency.json"
    _write(out, {"zen": {"hy3": {"n": 3, "p50_ms": 10, "p95_ms": 12, "last": TS}}})
    before = out.read_text(encoding="utf-8")

    fixture = tmp_path / "empty.jsonl"
    fixture.write_text(
        _new_wrap(event="provider_error") + "\n" + "garbage\n", encoding="utf-8"
    )
    assert feedback.main(["--json-file", str(fixture), "--out", str(out)]) == 0

    assert out.read_text(encoding="utf-8") == before
    assert "WARNING" in capsys.readouterr().err


def test_empty_input_writes_fresh_empty_file(feedback, tmp_path):
    out = tmp_path / "relay-latency.json"
    fixture = tmp_path / "empty.jsonl"
    fixture.write_text("garbage\n", encoding="utf-8")

    assert feedback.main(["--json-file", str(fixture), "--out", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["providers"] == {}
    assert payload["window"] == {"first": None, "last": None}
    assert out.read_text(encoding="utf-8").endswith("\n")


# ── select_dynamic_chain consumes the file ───────────────────────────────────


def _stable(providers):
    return {"providers": {k: {"models": v} for k, v in providers.items()}}


def _avail(providers):
    return {"providers": {k: v for k, v in providers.items()}}


HEALTHY = {"uptime_7d": 0.9, "uptime_30d": 0.9, "samples_7d": 10}


def _write_latency(path, generated_at, zen_p50, nvidia_p50, n=3):
    _write(
        path,
        {
            "zen": {"hy3": {"n": n, "p50_ms": zen_p50, "p95_ms": zen_p50, "last": TS}},
            "nvidia": {"hy3": {"n": n, "p50_ms": nvidia_p50, "p95_ms": nvidia_p50, "last": TS}},
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generated_at_utc"] = generated_at.isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")


def _group_chain(gs, monkeypatch, tmp_path, generated_at, zen_p50, nvidia_p50, n=3):
    path = tmp_path / "relay-latency.json"
    _write_latency(path, generated_at, zen_p50, nvidia_p50, n=n)
    monkeypatch.setattr(gs, "RELAY_LATENCY_PATH", path)
    stable = _stable({"zen": ["hy3"], "nvidia": ["hy3"]})
    avail = _avail({"zen": {"hy3": HEALTHY}, "nvidia": {"hy3": HEALTHY}})
    return [e["id"] for e in gs.select_dynamic_chain({"hy3": 100}, avail, stable)]


def test_fresh_latency_overrides_static_preference(gs, monkeypatch, tmp_path):
    """nvidia has the better real p50, so it leads despite zen preference."""
    now = datetime.now(timezone.utc)
    ids = _group_chain(gs, monkeypatch, tmp_path, now, zen_p50=900, nvidia_p50=100)
    assert ids == ["nvidia-hy3", "zen-hy3"]


def test_stale_latency_falls_back_to_preference(gs, monkeypatch, tmp_path):
    generated = datetime.now(timezone.utc) - timedelta(hours=25)
    ids = _group_chain(gs, monkeypatch, tmp_path, generated, zen_p50=900, nvidia_p50=100)
    assert ids == ["zen-hy3", "nvidia-hy3"]  # zen=0 preference wins


def test_thin_latency_falls_back_to_preference(gs, monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    ids = _group_chain(gs, monkeypatch, tmp_path, now, zen_p50=900, nvidia_p50=100, n=2)
    assert ids == ["zen-hy3", "nvidia-hy3"]


def test_missing_latency_file_uses_availability_order(gs, monkeypatch, tmp_path):
    monkeypatch.setattr(gs, "RELAY_LATENCY_PATH", tmp_path / "missing.json")
    stable = _stable({"zen": ["hy3"], "nvidia": ["hy3"]})
    avail = _avail(
        {
            "zen": {"hy3": {"uptime_7d": 0.5, "uptime_30d": 0.5, "samples_7d": 10}},
            "nvidia": {"hy3": HEALTHY},
        }
    )
    ids = [e["id"] for e in gs.select_dynamic_chain({"hy3": 100}, avail, stable)]
    assert ids == ["nvidia-hy3", "zen-hy3"]  # availability DESC, unchanged
