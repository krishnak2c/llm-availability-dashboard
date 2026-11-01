#!/usr/bin/env python3
"""Aggregate REAL relay latency from Vercel function logs.

The dashboard publishes a failover chain for the sibling relay repo. 1-token
availability probes cannot distinguish providers that are equally "up" but
wildly different in real-traffic latency (2026-11-01: Nemotron 550B p50 545ms
via openrouter vs 3670ms via NVIDIA, while probes measured 666ms vs 778ms).
This script turns relay production log lines into docs/relay-latency.json so
generate_site.py can order same-model provider groups by measured p50.

Input: JSONL on stdin (as `vercel logs --json` emits) or --json-file PATH.
Both the new CLI shape {"type": "log", "data": {...}} and the legacy shape
(data object at the top level) are accepted. No network calls are made.

Only `provider_success` events are counted; unparseable lines are skipped.
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT = Path("docs") / "relay-latency.json"

# Fields inside the relay's structured log message, e.g.
# "[relay] req=… event=provider_success provider=nvidia-… model=nvidia/…
#  status=200 retry=0 latency=3670ms"
_EVENT_RE = re.compile(r"event=(\S+)")
_PROVIDER_RE = re.compile(r"provider=(\S+)")
_MODEL_RE = re.compile(r"model=(\S+)")
_LATENCY_RE = re.compile(r"latency=(\d+(?:\.\d+)?)ms")

# Log timestamp field names seen across vercel CLI / function log shapes.
_TS_KEYS = ("timestamp", "time", "created", "createdAt", "date")


def _parse_timestamp(value):
    """Best-effort timestamp → aware UTC datetime, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000.0 if value > 1e11 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return _parse_timestamp(int(text))
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _line_timestamp(top):
    for key in _TS_KEYS:
        ts = _parse_timestamp(top.get(key))
        if ts is not None:
            return ts
    return None


def parse_log_line(line):
    """Parse one JSONL log line into a provider_success record, or None.

    Returns {"event", "provider", "model", "latency_ms", "timestamp"} where
    timestamp is an aware UTC datetime or None. Only provider_success lines
    are returned; everything else (other events, junk, non-JSON) is None.
    """
    line = (line or "").strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    # New CLI: {"type": "log", "data": {...}}; legacy: fields at top level.
    data = obj.get("data")
    if not isinstance(data, dict):
        data = obj

    message = data.get("message")
    if not isinstance(message, str):
        message = ""

    event_match = _EVENT_RE.search(message)
    if not event_match or event_match.group(1) != "provider_success":
        return None
    provider_match = _PROVIDER_RE.search(message)
    model_match = _MODEL_RE.search(message)
    latency_match = _LATENCY_RE.search(message)
    if not (provider_match and model_match and latency_match):
        return None

    latency = float(latency_match.group(1))
    return {
        "event": "provider_success",
        "provider": provider_match.group(1),
        "model": model_match.group(1),
        "latency_ms": latency,
        "timestamp": _line_timestamp(data) or _line_timestamp(obj),
    }


def percentile(sorted_values, pct):
    """Nearest-rank percentile of an already-sorted non-empty list."""
    if not sorted_values:
        return 0
    rank = math.ceil(pct / 100.0 * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def build_report(records, now=None):
    """Aggregate parsed records into the relay-latency.json payload."""
    now = now or datetime.now(timezone.utc)
    latencies = {}  # (provider, model) -> [ms, ...]
    last_seen = {}  # (provider, model) -> datetime
    window_first = None
    window_last = None

    for rec in records:
        key = (rec["provider"], rec["model"])
        latencies.setdefault(key, []).append(rec["latency_ms"])
        ts = rec.get("timestamp")
        if ts is not None:
            if key not in last_seen or ts > last_seen[key]:
                last_seen[key] = ts
            if window_first is None or ts < window_first:
                window_first = ts
            if window_last is None or ts > window_last:
                window_last = ts

    providers = {}
    for (provider, model), values in latencies.items():
        ordered = sorted(values)
        record = {
            "n": len(ordered),
            "p50_ms": int(round(percentile(ordered, 50))),
            "p95_ms": int(round(percentile(ordered, 95))),
            "last": (last_seen.get((provider, model)) or now).isoformat(),
        }
        providers.setdefault(provider, {})[model] = record

    return {
        "generated_at_utc": now.isoformat(),
        "window": {
            "first": window_first.isoformat() if window_first else None,
            "last": window_last.isoformat() if window_last else None,
        },
        "providers": providers,
    }


def _existing_has_data(path):
    """True when path exists and contains at least one provider record."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and bool(payload.get("providers"))


def write_report(path, report):
    """Write report atomically, guarding against replacing data with nothing.

    Mirrors write_stable_models: a zero-success run must never wipe a good
    relay-latency.json (log collection can legitimately return nothing).
    Returns True when the file was written, False when the old file was kept.
    """
    path = Path(path)
    if not report.get("providers") and _existing_has_data(path):
        print(
            "WARNING: 0 provider_success entries parsed — keeping the previous "
            f"{path} unchanged",
            file=sys.stderr,
        )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return True


def collect(lines):
    """Parse log lines into provider_success records."""
    records = []
    for line in lines:
        rec = parse_log_line(line)
        if rec:
            records.append(rec)
    return records


def run(lines, out_path=DEFAULT_OUT, now=None):
    """Parse log lines, aggregate, and write. Returns the report."""
    report = build_report(collect(lines), now=now)
    write_report(out_path, report)
    return report


def _print_summary(records):
    """One summary line per provider: p50/p95/n/max of its real latencies."""
    by_provider = {}
    for rec in records:
        by_provider.setdefault(rec["provider"], []).append(rec["latency_ms"])
    for provider in sorted(by_provider):
        ordered = sorted(by_provider[provider])
        print(
            f"{provider}: p50={int(round(percentile(ordered, 50)))}ms "
            f"p95={int(round(percentile(ordered, 95)))}ms "
            f"n={len(ordered)} max={int(round(ordered[-1]))}ms"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate real relay latency from Vercel logs (JSONL)."
    )
    parser.add_argument(
        "--json-file",
        metavar="PATH",
        help="read JSONL from PATH instead of stdin",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=str(DEFAULT_OUT),
        help=f"output path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)

    try:
        if args.json_file:
            with open(args.json_file, encoding="utf-8") as fh:
                lines = fh.readlines()
        else:
            lines = sys.stdin.readlines()
    except OSError as exc:
        print(f"ERROR: cannot read input: {exc}", file=sys.stderr)
        return 1

    records = collect(lines)
    report = build_report(records)
    try:
        write_report(args.out, report)
    except OSError as exc:
        print(f"ERROR: cannot write {args.out}: {exc}", file=sys.stderr)
        return 1

    _print_summary(records)
    print(f"Written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
