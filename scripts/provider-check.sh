#!/usr/bin/env bash
# =============================================================================
# Free-LLM availability dashboard — provider connectivity checks
# =============================================================================
# Hits each provider's /models (or equivalent) endpoint directly with the
# configured key to classify reachability. Mirrors the providers in
# probe_models.py; exit 2 only on AUTH_ERROR (actionable: fix the key).
#
# Classification:
#   OK            -> 200 with a JSON model list
#   SKIPPED       -> env key not set (provider intentionally unused)
#   AUTH_ERROR    -> 401/403 (key invalid / revoked)
#   RATE_LIMITED  -> 429 (valid key, throttled — NOT fatal)
#   UNREACHABLE   -> network error / timeout (provider down or DNS fail)
#   SERVER_ERROR  -> 5xx
#
# Usage:  ./scripts/provider-check.sh
# =============================================================================
set -uo pipefail

# shellcheck disable=SC1091
[[ -f ./.env ]] && { set -a; . ./.env; set +a; }

TIMEOUT=15
OK=0; SKIPPED=0; AUTH=0; RATE=0; UNREACH=0; SERVER=0

probe() {
  # $1 = provider label
  # $2 = url
  # $3 = auth header value (e.g. "Authorization: Bearer $KEY" or "x-goog-api-key: $KEY")
  # $4 = env var name holding the key (to detect "not configured")
  # $5 = 'optional' to allow probing anonymously when the key is empty
  local label="$1" url="$2" hdr="$3" envvar="$4" optional="${5:-required}"
  local key="${!envvar:-}"

  if [[ -z "$key" ]]; then
    if [[ "$optional" == "optional" ]]; then
      : # probe anonymously below
    else
      echo "  [SKIPPED]  $label  ($envvar not set)"
      SKIPPED=$((SKIPPED+1)); return
    fi
  fi

  local code body
  if [[ -n "$key" ]]; then
    code=$(curl -s -o /tmp/provider_probe.json -w '%{http_code}' --max-time "$TIMEOUT" \
      -H "$hdr" "$url" 2>/dev/null || echo "000")
  else
    code=$(curl -s -o /tmp/provider_probe.json -w '%{http_code}' --max-time "$TIMEOUT" "$url" 2>/dev/null || echo "000")
  fi
  body="$(cat /tmp/provider_probe.json 2>/dev/null | head -c 300)"

  case "$code" in
    200)
      if echo "$body" | grep -qiE '"models"|"data"|"id"' ; then
        echo "  [OK]       $label  -> $code"; OK=$((OK+1))
      else
        echo "  [WARN]     $label  -> $code but body looks unusual: ${body:0:80}"
        OK=$((OK+1))
      fi
      ;;
    401|403)
      echo "  [AUTH]     $label  -> $code (invalid key)"; AUTH=$((AUTH+1)) ;;
    429)
      echo "  [RATE]     $label  -> 429 (rate limited, not fatal)"; RATE=$((RATE+1)) ;;
    000)
      echo "  [UNREACH]  $label  -> timeout/network error"; UNREACH=$((UNREACH+1)) ;;
    5*)
      echo "  [5xx]      $label  -> $code"; SERVER=$((SERVER+1)) ;;
    *)
      echo "  [?]        $label  -> $code  ${body:0:60}"; SERVER=$((SERVER+1)) ;;
  esac
}

echo "== Free-LLM availability dashboard provider checks =="
echo

# Drive the probes from the single providers.py registry. providers.py emits
# one line per provider: label|models_url|header|env_var|optional.
# header carries a literal "$KEY" token that is swapped for the real key below.
while IFS='|' read -r label url hdr envvar optional; do
  [[ -z "$label" ]] && continue

  # Cloudflare needs an account id substituted into the URL
  if [[ "$url" == *'{account}'* ]]; then
    if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
      echo "  [SKIPPED]  $label  (CLOUDFLARE_API_KEY/ACCOUNT_ID not set)"; SKIPPED=$((SKIPPED+1))
      continue
    fi
    url="${url//\{account\}/$CLOUDFLARE_ACCOUNT_ID}"
  fi

  key="${!envvar:-}"
  if [[ -n "$key" ]]; then
    hdr="${hdr//\$\{KEY\}/$key}"
  fi
  probe "$label" "$url" "$hdr" "$envvar" "$optional"
done < <(python3 providers.py)

# Legacy anonymous check: pollinations stays out of probes + site (disabled),
# but keep the connectivity line so keyless reachability is still visible.
probe "Pollinations" "https://gen.pollinations.ai/v1/models" "" "POLLINATIONS_API_KEY" "optional"

echo
echo "== provider result: OK=$OK SKIPPED=$SKIPPED AUTH=$AUTH RATE=$RATE UNREACHABLE=$UNREACH SERVER=$SERVER =="
rm -f /tmp/provider_probe.json
# AUTH errors are actionable (fix the key); everything else is informational.
[[ "$AUTH" -eq 0 ]] || exit 2
exit 0
