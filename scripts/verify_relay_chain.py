#!/usr/bin/env python3
"""Fail the exit code when a providers.json would republish a dead relay chain.

2026-10-28 incident: the cron's publish step copied docs/providers.json into
the relay repo unconditionally while the file held a chain whose every entry
was dead upstream, which took the relay down for two days. This script is the
check that runs BEFORE that copy: it applies the same liveness rules as
generate_site.py's gate (imported, never duplicated) to a providers.json we
are about to publish and exits non-zero when the chain must not ship.

Usage:
    python3 scripts/verify_relay_chain.py [providers.json]   # default docs/providers.json

Exit codes: 0 publishable, 1 do not publish.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from generate_site import (  # noqa: E402
    RELAY_MIN_LIVE_ENTRIES,
    _relay_entry_dead_reason,
)

DEFAULT_PROVIDERS = REPO_ROOT / "docs" / "providers.json"
DEFAULT_AVAIL = REPO_ROOT / "docs" / "availability.json"


def verify(providers_path: Path, avail_path: Path) -> tuple[bool, list[str]]:
    """(publishable, report_lines) for one providers.json against availability data."""
    try:
        payload = json.loads(providers_path.read_text(encoding="utf-8"))
    except Exception as exc:  # unparseable artifact → never publish it
        return False, [f"UNREADABLE {providers_path}: {exc}"]
    try:
        avail = json.loads(avail_path.read_text(encoding="utf-8")).get("providers", {})
    except Exception:
        avail = {}

    chain = payload.get("chain") or []
    if not chain:
        return False, ["EMPTY chain: nothing to publish"]

    lines: list[str] = []
    dead = 0
    for entry in chain:
        reason = _relay_entry_dead_reason(entry, avail)
        if reason:
            dead += 1
            lines.append(f"DEAD {entry.get('id') or entry.get('model') or '?'}: {reason}")

    live = len(chain) - dead
    lines.append(
        f"{live} live / {len(chain)} entries"
        + (f" (minimum {RELAY_MIN_LIVE_ENTRIES})" if live < len(chain) else "")
    )
    if dead:
        return False, lines
    return True, lines


def main(argv: list[str]) -> int:
    providers_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PROVIDERS
    avail_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_AVAIL
    ok, lines = verify(providers_path, avail_path)
    for line in lines:
        print(line)
    if not ok:
        print(
            "Relay chain failed liveness check — not publishing; the relay keeps its current chain."
        )
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
