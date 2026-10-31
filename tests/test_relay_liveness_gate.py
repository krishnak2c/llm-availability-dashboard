"""Tests for the relay liveness gate in generate_site.py.

providers.json is the relay's failover order. Before this gate existed, a
stale static RELAY_CHAIN with zero live entries was published verbatim and the
relay served HTTP 410 to every chat request for two days (2026-10-28).

Invariants under test:
- an entry is published only when availability.json proves it alive
  (no record == unknown == dead)
- `order` always matches the surviving chain
- fewer than RELAY_MIN_LIVE_ENTRIES survivors → the previous providers.json is
  left untouched and a WARNING names the survivor count
- output stays byte-compatible with the relay's expected shape
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gs():
    return load_module("generate_site", "generate_site.py")


HEALTHY = {
    "uptime_7d": 1.0,
    "uptime_30d": 1.0,
    "samples_7d": 24,
    "last_status": "ok",
}
FLAKY_BUT_ALIVE = {
    "uptime_7d": 0.8,
    "uptime_30d": 0.9,
    "samples_7d": 10,
    "last_status": "ok",
}
DEAD_QUOTA = {"uptime_7d": 0.0, "uptime_30d": 0.2, "samples_7d": 6, "last_status": "quota_exhausted"}
DEAD_UPTIME = {"uptime_7d": 0.0, "uptime_30d": 0.1, "samples_7d": 4, "last_status": "timeout"}
ONE_GOOD_PROBE = {"uptime_7d": None, "uptime_30d": None, "samples_7d": 0, "last_status": "ok"}


def _all_dead_avail(gs):
    """Every model the static RELAY_CHAIN references, recorded dead."""
    avail = {}
    for _model, (prov_key, model_key) in gs.RELAY_AVAIL_MAP.items():
        avail.setdefault(prov_key, {})[model_key] = dict(DEAD_QUOTA)
    return avail


def _chain_entry(prov_key, model, norm=None):
    """A dynamic-style chain entry (id = <provider>-<normalized model>)."""
    return {
        "id": f"{prov_key}-{norm or model}",
        "baseUrl": f"https://{prov_key}.example/v1",
        "model": model,
        "apiKeyEnv": "TEST_KEY",
    }


def _payload(entries, allowlist=None):
    return {
        "order": [e["id"] for e in entries],
        "chain": entries,
        "modelAllowlist": [e["model"] for e in entries]
        if allowlist is None
        else allowlist,
    }


def _previous_file(tmp_path):
    """A sentinel providers.json that must survive a refused publish."""
    path = tmp_path / "providers.json"
    previous = {
        "order": ["old-a"],
        "chain": [{"id": "old-a", "baseUrl": "https://old/v1", "model": "old-a", "apiKeyEnv": "K"}],
        "modelAllowlist": ["old-a"],
    }
    raw = json.dumps(previous, indent=2, ensure_ascii=False)
    path.write_text(raw, encoding="utf-8")
    return path, raw


# ── (a) every entry dead → previous providers.json preserved ─────────────────


def test_all_entries_dead_preserves_previous_file(gs, tmp_path, capsys):
    avail = _all_dead_avail(gs)
    payload = gs.build_relay_providers_json(avail)  # static fallback path
    assert payload["chain"], "static fallback must produce candidates to gate"
    path, raw_before = _previous_file(tmp_path)

    published = gs.write_relay_payload(tmp_path, payload, avail)

    assert published is None
    assert path.read_text(encoding="utf-8") == raw_before  # untouched
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "0 live entries" in out
    # every candidate was logged with a reason
    for entry in payload["chain"]:
        assert f"dropped {entry['id']}" in out


def test_all_entries_unprobed_also_refuses_publish(gs, tmp_path, capsys):
    """No availability record at all is unknown, not healthy."""
    payload = gs.build_relay_providers_json({})
    path, raw_before = _previous_file(tmp_path)

    assert gs.write_relay_payload(tmp_path, payload, {}) is None
    assert path.read_text(encoding="utf-8") == raw_before
    assert "no availability record" in capsys.readouterr().out


# ── (b) mixed live/dead → dead dropped, live kept, order recomputed ──────────


def test_mixed_chain_drops_dead_and_reorders(gs, tmp_path, capsys):
    entries = [
        _chain_entry("zen", "alpha"),  # live
        _chain_entry("zen", "dead-quota"),  # dropped: status
        _chain_entry("zen", "dead-zero"),  # dropped: uptime_7d == 0, n=4
        _chain_entry("nvidia", "beta"),  # live
        _chain_entry("nvidia", "gamma"),  # live
        _chain_entry("kilo", "delta"),  # live
    ]
    avail = {
        "zen": {
            "alpha": HEALTHY,
            "dead-quota": DEAD_QUOTA,
            "dead-zero": DEAD_UPTIME,
        },
        "nvidia": {"beta": FLAKY_BUT_ALIVE, "gamma": HEALTHY},
        "kilo": {"delta": HEALTHY},
    }
    payload = _payload(entries)

    published = gs.write_relay_payload(tmp_path, payload, avail)

    assert published is not None
    survivors = ["zen-alpha", "nvidia-beta", "nvidia-gamma", "kilo-delta"]
    assert published["order"] == survivors
    assert [e["id"] for e in published["chain"]] == survivors
    # allowlist pruned to models still referenced
    assert published["modelAllowlist"] == ["alpha", "beta", "gamma", "delta"]
    assert "dead-quota" not in json.dumps(published)

    out = capsys.readouterr().out
    assert "dropped zen-dead-quota: last_status=quota_exhausted" in out
    assert "dropped zen-dead-zero: uptime_7d=0 over 4 samples" in out

    # byte-compatible on-disk shape: same keys, indent=2, ensure_ascii=False
    on_disk = (tmp_path / "providers.json").read_text(encoding="utf-8")
    assert on_disk == json.dumps(published, indent=2, ensure_ascii=False)
    assert list(json.loads(on_disk)) == ["order", "chain", "modelAllowlist"]


# ── (c) entry with no availability record → dropped ─────────────────────────


def test_entry_missing_from_availability_is_dropped(gs, tmp_path, capsys):
    entries = [
        _chain_entry("zen", "alpha"),
        _chain_entry("nvidia", "beta"),
        _chain_entry("kilo", "delta"),
        _chain_entry("llm7", "eps"),
        _chain_entry("llm7", "ghost-model"),  # no record anywhere
    ]
    avail = {
        "zen": {"alpha": HEALTHY},
        "nvidia": {"beta": HEALTHY},
        "kilo": {"delta": HEALTHY},
        "llm7": {"eps": HEALTHY},
    }

    published = gs.write_relay_payload(tmp_path, _payload(entries), avail)

    assert published is not None
    assert "llm7-ghost-model" not in published["order"]
    assert "ghost-model" not in published["modelAllowlist"]
    assert "dropped llm7-ghost-model: no availability record" in capsys.readouterr().out


# ── (d) samples_7d == 0 with last_status == "ok" → kept ─────────────────────


def test_single_ok_probe_with_zero_samples_is_kept(gs):
    entry = _chain_entry("zen", "alpha")
    assert gs._relay_entry_dead_reason(entry, {"zen": {"alpha": ONE_GOOD_PROBE}}) is None

    entries = [entry, _chain_entry("nvidia", "beta"), _chain_entry("kilo", "delta"),
               _chain_entry("llm7", "eps")]
    avail = {
        "zen": {"alpha": ONE_GOOD_PROBE},
        "nvidia": {"beta": ONE_GOOD_PROBE},
        "kilo": {"delta": ONE_GOOD_PROBE},
        "llm7": {"eps": ONE_GOOD_PROBE},
    }
    kept, kept_ids, dropped = gs.apply_relay_liveness_gate(_payload(entries), avail)
    assert dropped == []
    assert kept_ids == ["zen-alpha", "nvidia-beta", "kilo-delta", "llm7-eps"]


# ── (e) fewer than 4 survivors → do not publish ─────────────────────────────


def test_fewer_than_min_live_entries_does_not_publish(gs, tmp_path, capsys):
    entries = [
        _chain_entry("zen", "alpha"),
        _chain_entry("nvidia", "beta"),
        _chain_entry("kilo", "delta"),
        _chain_entry("llm7", "dead1"),
        _chain_entry("llm7", "dead2"),
    ]
    avail = {
        "zen": {"alpha": HEALTHY},
        "nvidia": {"beta": HEALTHY},
        "kilo": {"delta": HEALTHY},
        "llm7": {"dead1": DEAD_QUOTA, "dead2": DEAD_UPTIME},
    }
    path, raw_before = _previous_file(tmp_path)

    published = gs.write_relay_payload(tmp_path, _payload(entries), avail)

    assert published is None
    assert path.read_text(encoding="utf-8") == raw_before
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "3 live entries" in out
    assert f"minimum {gs.RELAY_MIN_LIVE_ENTRIES}" in out


def test_exactly_min_live_entries_publishes(gs, tmp_path):
    entries = [
        _chain_entry("zen", "alpha"),
        _chain_entry("nvidia", "beta"),
        _chain_entry("kilo", "delta"),
        _chain_entry("llm7", "eps"),
        _chain_entry("llm7", "dead1"),
    ]
    avail = {
        "zen": {"alpha": HEALTHY},
        "nvidia": {"beta": HEALTHY},
        "kilo": {"delta": HEALTHY},
        "llm7": {"eps": HEALTHY, "dead1": DEAD_QUOTA},
    }

    published = gs.write_relay_payload(tmp_path, _payload(entries), avail)

    assert published is not None
    assert published["order"] == ["zen-alpha", "nvidia-beta", "kilo-delta", "llm7-eps"]


# ── static-fallback coverage (the actual incident path) ─────────────────────


def test_static_fallback_chain_is_gated(gs, tmp_path):
    """Four healthy static models, the rest dead → only the four ship."""
    live_models = ["deepseek-v4-flash-free", "hy3-free", "nemotron-3-ultra-free", "mimo-v2.5-free"]
    avail = {}
    for model, (prov_key, model_key) in gs.RELAY_AVAIL_MAP.items():
        stats = HEALTHY if model in live_models else DEAD_QUOTA
        avail.setdefault(prov_key, {})[model_key] = dict(stats)

    payload = gs.build_relay_providers_json(avail)
    published = gs.write_relay_payload(tmp_path, payload, avail)

    assert published is not None
    assert sorted(e["model"] for e in published["chain"]) == sorted(live_models)
    assert published["order"] == [e["id"] for e in published["chain"]]
    # dead static models pruned from the allowlist, excluded-provider models absent
    for dead in ("z-ai/glm-5.2", "minimaxai/minimax-m3", "thinkingmachines/inkling"):
        assert dead not in published["modelAllowlist"]
    assert "qwen/qwen3.6-27b" not in published["modelAllowlist"]


def test_dynamic_entry_resolves_stats_by_id_provider(gs):
    """Dynamic ids encode their provider; lookup must use it, not RELAY_AVAIL_MAP."""
    entry = {"id": "zen-somemodel", "baseUrl": "u", "model": "somemodel", "apiKeyEnv": "K"}
    assert gs._relay_entry_stats(entry, {"zen": {"somemodel": HEALTHY}}) == HEALTHY
    assert gs._relay_entry_stats(entry, {"zen": {"nope": HEALTHY}}) is None
    # Provider-exact: the same model alive on ANOTHER provider proves nothing
    # about this entry's baseUrl.
    assert gs._relay_entry_stats(entry, {"other": {"somemodel": HEALTHY}}) is None


# ── Task 1: stable_models.json must never be emptied in place ────────────────
# An empty stable set is the first domino of the 2026-10-28 incident: it makes
# hermes_rank.py emit `ranking: []`, which makes the site fall back to the
# stale static RELAY_CHAIN.


def _stable_payload(providers):
    return {"providers": {k: {"models": v} for k, v in providers.items()}}


def test_stable_models_guard_keeps_previous_file(gs, tmp_path, capsys):
    path = tmp_path / "stable_models.json"
    previous = json.dumps(
        _stable_payload({"zen": [{"id": "a"}, {"id": "b"}]}), indent=2
    )
    path.write_text(previous, encoding="utf-8")

    written, n = gs.write_stable_models(path, {"providers": {}})

    assert written is False
    assert n == 0
    assert path.read_text(encoding="utf-8") == previous
    out = capsys.readouterr().out
    assert "WARNING" in out and "2 models" in out


def test_stable_models_guard_writes_when_no_previous(gs, tmp_path):
    path = tmp_path / "stable_models.json"
    written, n = gs.write_stable_models(path, {"providers": {}})
    assert written is True and n == 0
    assert json.loads(path.read_text()) == {"providers": {}}


def test_stable_models_guard_writes_non_empty_set(gs, tmp_path):
    path = tmp_path / "stable_models.json"
    stable = _stable_payload({"zen": [{"id": "a"}]})
    written, n = gs.write_stable_models(path, stable)
    assert written is True and n == 1
    assert json.loads(path.read_text()) == stable
