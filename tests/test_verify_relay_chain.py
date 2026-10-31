"""scripts/verify_relay_chain.py must refuse to bless an un-publishable relay chain.

2026-10-28 incident: the update-models cron copied a providers.json whose every
entry was dead upstream into the relay repo, taking the relay down for two days.
These tests pin the verify-before-publish exit contract.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "verify_relay_chain", REPO / "scripts" / "verify_relay_chain.py"
)
verify_relay_chain = importlib.util.module_from_spec(SPEC)
sys.modules["verify_relay_chain"] = verify_relay_chain
SPEC.loader.exec_module(verify_relay_chain)

verify = verify_relay_chain.verify
AVAIL = REPO / "docs" / "availability.json"


def _entry(pid: str, model: str) -> dict:
    return {
        "id": pid,
        "baseUrl": f"https://example.test/{pid}/v1",
        "model": model,
        "apiKeyEnv": f"{pid.upper()}_API_KEY",
    }


def _chain(*entries: dict) -> dict:
    e = list(entries)
    return {"order": [x["id"] for x in e], "chain": e, "modelAllowlist": [x["model"] for x in e]}


def _avail(unproven_model: str | None = None, n_live: int = 0) -> dict:
    """Availability where the first n_live 'demo/chat' slots are probed ok."""
    providers: dict = {}
    demos = {}
    for i in range(n_live):
        demos[f"demo/chat-{i}"] = {"uptime_7d": 1.0, "uptime_30d": 1.0, "samples_7d": 6,
                                    "last_status": "ok", "hourly_uptime": []}
    providers["demo"] = demos
    if unproven_model:
        providers["demo"] = {**demos, unproven_model: {
            "uptime_7d": 0.0, "uptime_30d": 0.0, "samples_7d": 6,
            "last_status": "rate_limited", "down_reason_7d": "rate_limited",
            "hourly_uptime": []}}
    return {"providers": providers}


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "providers.json"
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p


def _write_av(tmp_path: Path, avail: dict) -> Path:
    p = tmp_path / "availability.json"
    p.write_text(json.dumps(avail), encoding="utf-8")
    return p


def test_deployed_chain_is_publishable(tmp_path=None):
    """The real restored docs/providers.json (in sync with the deployed relay) passes."""
    ok, lines = verify(REPO / "docs" / "providers.json", AVAIL)
    assert ok, "\n".join(lines)


def test_all_dead_chain_is_rejected(tmp_path):
    ap = _write_av(tmp_path, _avail())
    pp = _write(tmp_path, _chain(
        _entry("demo-a", "demo/dead-1"),
    ))
    ok, lines = verify(pp, ap)
    assert not ok and any("no availability record" in x or "DEAD" in x for x in lines), lines


def test_dead_status_entry_is_named(tmp_path):
    ap = _write_av(tmp_path, _avail(unproven_model="demo/rate-limited"))
    pp = _write(tmp_path, _chain(_entry("demo-x", "demo/rate-limited")))
    ok, lines = verify(pp, ap)
    assert not ok
    assert any("DEAD demo-x" in line and "rate_limited" in line for line in lines), lines


def test_missing_availability_record_is_rejected(tmp_path):
    ap = _write_av(tmp_path, _avail())
    pp = _write(tmp_path, _chain(_entry("ghost-y", "demo/never-probed")))
    ok, lines = verify(pp, ap)
    assert not ok and any("no availability record" in line for line in lines), lines


def test_unparseable_providers_file_is_rejected(tmp_path):
    ap = _write_av(tmp_path, _avail())
    pp = tmp_path / "providers.json"
    pp.write_text("{not json", encoding="utf-8")
    ok, lines = verify(pp, ap)
    assert not ok and any("UNREADABLE" in line for line in lines), lines


def test_missing_chain_is_rejected(tmp_path):
    ap = _write_av(tmp_path, _avail())
    pp = _write(tmp_path, {"order": [], "chain": [], "modelAllowlist": []})
    ok, lines = verify(pp, ap)
    assert not ok and any("EMPTY" in line for line in lines), lines
