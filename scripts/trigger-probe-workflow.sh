#!/usr/bin/env bash
# =============================================================================
# Free-LLM availability dashboard — dispatch probe.yml from an external cron
# =============================================================================
# The VM's ONLY job is to poke GitHub: POST a workflow_dispatch. All probing,
# committing and pushing stays inside GitHub Actions (`probe.yml`), so this
# script needs no provider keys, no clone, no push rights — just a token that
# can dispatch.
#
# Why: GitHub's `schedule` trigger is best-effort — runs get dropped under
# load, and top-of-hour crons hit the worst window. A VM cron has no scheduler
# in the middle, so the cadence is exact. This replaces external cron-job.org
# (which also had to be trusted with a token).
#
# Usage:
#   ./scripts/trigger-probe-workflow.sh            # dispatch (POST)
#   ./scripts/trigger-probe-workflow.sh --verify   # can this token see it? (GET)
#
# Token resolution — first non-empty wins:
#   1. $GITHUB_DISPATCH_TOKEN
#   2. token file, default ~/.config/llm-dashboard/dispatch.token
#      (override: $DISPATCH_TOKEN_FILE) — file holds just the token
#   3. `gh auth token`, only if the gh CLI is installed
# The token is never printed; if it shows up in an error body it is redacted.
#
# Config (env vars, all optional):
#   GITHUB_REPO          default krishnak2c/llm-availability-dashboard
#   GITHUB_WORKFLOW      default probe.yml
#   GITHUB_REF           default main
#   DISPATCH_TOKEN_FILE  default ~/.config/llm-dashboard/dispatch.token
#   PROBE_LOG_FILE       append the result line here (default: no log file)
#
# Exit codes:
#   0 = dispatched (HTTP 200 or 204) / token verified
#   1 = no token, auth/HTTP failure, or network failure after retries
#   2 = bad CLI usage
#
# Notes:
#   * Success = 204 (older API) or 200 (newer API, returns workflow_run_id).
#   * curl connection errors / timeouts are retried 3x: 5s, 15s, 45s.
#     4xx is NOT retried — a 401/403 is a config problem, retrying won't help.
#   * No local concurrency guard: `probe.yml` already has
#     `concurrency: group: probe-models`, and a dispatch is cheap.
# =============================================================================
set -uo pipefail

GITHUB_REPO="${GITHUB_REPO:-krishnak2c/llm-availability-dashboard}"
GITHUB_WORKFLOW="${GITHUB_WORKFLOW:-probe.yml}"
GITHUB_REF="${GITHUB_REF:-main}"
DISPATCH_TOKEN_FILE="${DISPATCH_TOKEN_FILE:-${HOME:-/root}/.config/llm-dashboard/dispatch.token}"

API_VERSION="2022-11-28"
MAX_TIME=20
# 3 retries -> 4 attempts total; these are the waits between them.
RETRY_DELAYS=(5 15 45)

TOKEN=""
TOKEN_SOURCE=""

# ── CLI ──────────────────────────────────────────────────────────────────────
MODE="dispatch"
case "${1:-}" in
  "")        MODE="dispatch" ;;
  --verify)  MODE="verify" ;;
  *)
    echo "usage: $(basename "$0") [--verify]" >&2
    echo "  (no args)  dispatch $GITHUB_WORKFLOW on $GITHUB_REPO (ref=$GITHUB_REF)" >&2
    echo "  --verify   check that the token can see the workflow (read-only)" >&2
    exit 2
    ;;
esac

# ── Token resolution ─────────────────────────────────────────────────────────
# Never echo the token. Sourced tokens are only ever placed in a curl header.
if [[ -n "${GITHUB_DISPATCH_TOKEN:-}" ]]; then
  TOKEN="$GITHUB_DISPATCH_TOKEN"
  TOKEN_SOURCE="env:GITHUB_DISPATCH_TOKEN"
elif [[ -f "$DISPATCH_TOKEN_FILE" ]]; then
  # $(<file) strips trailing newline(s); also trim CR from CRLF files.
  TOKEN="$(tr -d '\r' < "$DISPATCH_TOKEN_FILE" 2>/dev/null)"
  [[ -n "$TOKEN" ]] && TOKEN_SOURCE="file:$DISPATCH_TOKEN_FILE"
fi

if [[ -z "$TOKEN" ]] && command -v gh >/dev/null 2>&1; then
  TOKEN="$(gh auth token 2>/dev/null)"
  if [[ -n "$TOKEN" ]]; then
    TOKEN_SOURCE="gh-cli"
    echo "note: no dispatch token configured, falling back to \`gh auth token\`" >&2
    echo "      (persist a dedicated token in $DISPATCH_TOKEN_FILE for cron)" >&2
  fi
fi

# printf, not cat: this path must work even with a minimal cron PATH.
if [[ -z "$TOKEN" ]]; then
  {
    printf '%s\n\n' "error: no GitHub token found — cannot dispatch $GITHUB_WORKFLOW."
    printf '%s\n' "Provide one of:"
    printf '%s\n' "  1. GITHUB_DISPATCH_TOKEN in the environment"
    printf '%s\n' "       export GITHUB_DISPATCH_TOKEN=...        # or set it in the crontab"
    printf '%s\n' "  2. a token file (default $DISPATCH_TOKEN_FILE, override with DISPATCH_TOKEN_FILE)"
    printf '%s\n' "       install -m 600 /dev/null \"$DISPATCH_TOKEN_FILE\""
    printf '%s\n' "       printf '%s' 'github_pat_...' > \"$DISPATCH_TOKEN_FILE\""
    printf '%s\n' "  3. an authenticated \`gh\` CLI (fallback, not recommended for cron)"
    printf '%s\n\n' "       gh auth login && gh auth token"
    printf '%s\n' "The token needs Actions: write on $GITHUB_REPO (classic PAT: \`repo\` + \`workflow\`)."
  } >&2
  exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
WORK="$(mktemp -d "${TMPDIR:-/tmp}/trigger-probe.XXXXXX")" || {
  echo "error: cannot create temp dir" >&2; exit 1
}
BODY_FILE="$WORK/body"
ERR_FILE="$WORK/curl.err"
HTTP_CODE="000"
CURL_RC=0
BODY=""
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

# Same line to stdout and (optionally) to the log file.
emit() {
  local line="$1"
  echo "$line"
  if [[ -n "${PROBE_LOG_FILE:-}" ]]; then
    mkdir -p "$(dirname "$PROBE_LOG_FILE")" 2>/dev/null
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$line" >> "$PROBE_LOG_FILE" 2>/dev/null ||
      echo "warn: cannot write $PROBE_LOG_FILE" >&2
  fi
}

# Belt and braces: strip the live token, then anything shaped like a PAT.
redact() {
  local s="$1"
  [[ -n "$TOKEN" ]] && s="${s//"$TOKEN"/***REDACTED***}"
  printf '%s' "$s" |
    sed -E 's/gh[pousr]_[A-Za-z0-9]{10,}/***REDACTED***/g; s/github_pat_[A-Za-z0-9_]{10,}/***REDACTED***/g'
}

# One curl attempt. Sets HTTP_CODE and CURL_RC (non-zero = network/timeout).
curl_once() {
  local method="$1" url="$2" data="${3:-}"
  local args=(
    -sS -o "$BODY_FILE" -w '%{http_code}' --max-time "$MAX_TIME"
    -X "$method"
    -H "Authorization: Bearer $TOKEN"
    -H "Accept: application/vnd.github+json"
    -H "X-GitHub-Api-Version: $API_VERSION"
  )
  [[ -n "$data" ]] && args+=(-H "Content-Type: application/json" --data "$data")

  : > "$BODY_FILE"; : > "$ERR_FILE"
  HTTP_CODE="$(curl "${args[@]}" "$url" 2>"$ERR_FILE")"
  CURL_RC=$?
  [[ -s "$BODY_FILE" ]] && BODY="$(cat "$BODY_FILE")" || BODY=""
}

# api_call METHOD URL [DATA] — retries network failures only, never 4xx.
api_call() {
  local method="$1" url="$2" data="${3:-}"
  local attempt=1 delay

  while :; do
    curl_once "$method" "$url" "$data"

    [[ "$CURL_RC" -eq 0 ]] && return 0

    if [[ "$attempt" -gt "${#RETRY_DELAYS[@]}" ]]; then
      local err="network error after $((attempt)) attempts (curl exit $CURL_RC)"
      [[ -s "$ERR_FILE" ]] && err+=": $(head -c 300 "$ERR_FILE")"
      echo "error: $err" >&2
      return 1
    fi

    delay="${RETRY_DELAYS[$((attempt - 1))]}"
    echo "warn: network error (curl exit $CURL_RC), retry $attempt/${#RETRY_DELAYS[@]} in ${delay}s" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

# Pull one top-level string field out of a JSON body (best effort, no jq dep).
json_field() {
  local key="$1"
  printf '%s' "$BODY" |
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get(sys.argv[1],""))' "$key" 2>/dev/null
}

# Print the failure body safely, truncated and redacted.
print_error_body() {
  local body="$1"
  body="$(printf '%s' "$body" | head -c 800)"
  echo "  response: $(redact "$body")" >&2
}

status_hint() {
  case "$1" in
    401) echo "  hint: token is invalid or expired" >&2 ;;
    403) echo "  hint: token lacks Actions: write on $GITHUB_REPO (classic PAT needs repo+workflow)" >&2 ;;
    404) echo "  hint: repo or workflow name wrong, or token cannot see the repo" >&2 ;;
    429) echo "  hint: GitHub rate-limited the token — back off before retrying" >&2 ;;
    5*)  echo "  hint: GitHub server error — safe to retry later" >&2 ;;
  esac
}

# ── --verify: read-only permission check ─────────────────────────────────────
if [[ "$MODE" == "verify" ]]; then
  URL="https://api.github.com/repos/$GITHUB_REPO/actions/workflows/$GITHUB_WORKFLOW"
  api_call GET "$URL" || exit 1

  if [[ "$HTTP_CODE" == "200" ]]; then
    emit "token OK: can access $GITHUB_WORKFLOW ($GITHUB_REPO)"
    name="$(json_field name)"
    path="$(json_field path)"
    [[ -n "$name" ]] && echo "  name: $name"
    [[ -n "$path" ]] && echo "  path: $path"
    echo "  token source: $TOKEN_SOURCE"
    exit 0
  fi

  echo "error: cannot verify $GITHUB_WORKFLOW on $GITHUB_REPO — HTTP $HTTP_CODE" >&2
  status_hint "$HTTP_CODE"
  print_error_body "$BODY"
  exit 1
fi

# ── Dispatch ─────────────────────────────────────────────────────────────────
URL="https://api.github.com/repos/$GITHUB_REPO/actions/workflows/$GITHUB_WORKFLOW/dispatches"
DATA="$(printf '{"ref":"%s"}' "$GITHUB_REF")"

api_call POST "$URL" "$DATA" || exit 1

case "$HTTP_CODE" in
  200|204)
    emit "dispatched $GITHUB_WORKFLOW on $GITHUB_REPO (ref=$GITHUB_REF, HTTP $HTTP_CODE)"
    run_id="$(json_field workflow_run_id)"
    [[ -n "$run_id" ]] && echo "  workflow_run_id: $run_id"
    exit 0
    ;;
  *)
    echo "error: dispatch of $GITHUB_WORKFLOW on $GITHUB_REPO failed — HTTP $HTTP_CODE" >&2
    status_hint "$HTTP_CODE"
    print_error_body "$BODY"
    exit 1
    ;;
esac
