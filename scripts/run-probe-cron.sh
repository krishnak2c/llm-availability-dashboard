#!/usr/bin/env bash
# =============================================================================
# Free-LLM availability dashboard — cron runner for probe_models.py
# =============================================================================
# Runs ONE probe round on a VM and publishes the result, doing everything the
# GitHub Actions `probe.yml` workflow did: probe -> stage docs/ -> commit ->
# push. Intended to be driven by the VM's cron (or a systemd timer).
#
# Why: GitHub's `schedule` trigger is best-effort — runs get dropped under
# load, and top-of-hour crons hit the worst window. A VM cron has no
# scheduler in the middle, so the cadence is exact.
#
# Usage:
#   ./scripts/run-probe-cron.sh            # probe + commit + push
#   ./scripts/run-probe-cron.sh --no-push  # probe + commit only (local dry run)
#
# Requirements on the VM:
#   * python3 with stdlib only (no pip install needed)
#   * a git clone of the dashboard repo with push rights
#     (SSH key or a fine-grained PAT with `Contents: Write`)
#   * a .env file with the provider keys you want probed
#     (see .env.example — every key is optional; providers without a key are
#      skipped, except llm7 which is probed anonymously)
#   * `docs/models.json` must exist — it is committed to the repo and
#     refreshed by update-models.yml. If it is missing, run
#     `python3 generate_site.py` once to create it.
#
# Exit codes: 0 = ok (probed, and pushed or --no-push), 1 = failure.
# =============================================================================
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PUSH=1
[[ "${1:-}" == "--no-push" ]] && PUSH=0

cd "$REPO_DIR" || { echo "error: cannot cd to $REPO_DIR" >&2; exit 1; }

# ── Load provider keys ───────────────────────────────────────────────────────
# shellcheck disable=SC1091
if [[ -f ./.env ]]; then
  set -a; . ./.env; set +a
else
  echo "warn: no .env in $REPO_DIR — probing anonymously (llm7 only)" >&2
fi

# ── Concurrency guard ────────────────────────────────────────────────────────
# A slow probe round must never overlap the next one. flock is released
# automatically when the process exits, including on crash.
# Repo-specific lock path so two different clones on one VM don't block
# each other.
LOCK="${PROBE_LOCK:-/tmp/llm-dashboard-probe-$(printf '%s' "$REPO_DIR" | md5sum | cut -c1-8).lock}"
exec 9>"$LOCK" || { echo "error: cannot open lock $LOCK" >&2; exit 1; }
if ! flock -n 9; then
  echo "skip: another probe round is still running (lock $LOCK)"
  exit 0
fi

# ── Probe ─────────────────────────────────────────────────────────────────────
# One round probes 1/4 of the catalog (ROUND_ROBIN_BUCKETS) plus every model
# on the watch list, so hourly runs give each model a probe roughly every 4h.
LOG_DIR="${PROBE_LOG_DIR:-$REPO_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/probe-$(date -u '+%Y-%m-%d').log"

echo "== probe round $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==" | tee -a "$LOG_FILE"
if ! python3 probe_models.py 2>&1 | tee -a "$LOG_FILE"; then
  echo "error: probe_models.py failed" | tee -a "$LOG_FILE" >&2
  exit 1
fi

# ── Publish ───────────────────────────────────────────────────────────────────
# docs/ is bot-generated: stage only the files a probe round writes.
for p in docs/probes.jsonl docs/availability.json docs/probes-archive; do
  [[ -e "$p" ]] && git add "$p"
done

if git diff --cached --quiet; then
  echo "no changes to commit"
  exit 0
fi

git config user.name  "llm-dashboard-cron"
git config user.email "cron@localhost"
git commit -m "chore: probe results $(date -u '+%Y-%m-%d %H:%M')" | tee -a "$LOG_FILE"

if [[ "$PUSH" -eq 0 ]]; then
  echo "committed locally (--no-push)"
  exit 0
fi

# Rebase-and-retry: another writer (the GitHub Actions backstop, or a second
# VM) may have pushed in the meantime.
for i in 1 2 3 4; do
  if git pull --rebase origin "${GIT_BRANCH:-main}" && git push origin "${GIT_BRANCH:-main}"; then
    echo "pushed to origin"
    exit 0
  fi
  echo "push attempt $i failed, retrying..." | tee -a "$LOG_FILE"
  sleep $((2 ** i))
done

echo "error: could not push after 4 attempts" | tee -a "$LOG_FILE" >&2
exit 1
