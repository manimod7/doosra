#!/usr/bin/env bash
# Checks a deployed Doosra end to end without writing anything (no bets, no logged prices).
#   API_URL=https://doosra-api-....run.app TOKEN=... [SITE_URL=https://....vercel.app SITE_PASSWORD=...] scripts/smoke.sh
# Runs from GitHub Actions ("Smoke test" workflow) or your own terminal. Needs curl and python3.
set -uo pipefail
: "${API_URL:?set API_URL}" "${TOKEN:?set TOKEN}"
API_URL=${API_URL%/}
fail=0
ok()  { echo "PASS  $1"; }
bad() { echo "FAIL  $1"; fail=1; }
get() { curl -sS --max-time 120 "$@"; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

h=$(get "$API_URL/health") && echo "$h" | json "d['status']" | grep -q ok && ok "API is up (first request can take ~30s while it starts)" || bad "API health: $h"

e=$(get "$API_URL/engine")
if v=$(echo "$e" | json "d['model']['version']" 2>/dev/null) && [ -n "$v" ] && [ "$v" != "None" ]; then
  ok "model loaded from the database: $v, learned up to $(echo "$e" | json "d['model']['trained_through']")"
else
  bad "no model yet: run the Learning cycle workflow once, then redeploy or wait for the next cold start"
fi

c=$(get "$API_URL/catalog")
n=$(echo "$c" | json "len(d['teams'])" 2>/dev/null || echo 0)
[ "${n:-0}" -gt 100 ] && ok "catalog: $n teams" || bad "catalog: $c"

body='{"team_a":"Mumbai Indians","team_b":"Chennai Super Kings","venue":"Wankhede Stadium","sims":500}'
card=$(get -H "Content-Type: application/json" -d "$body" "$API_URL/markets/card")
if w=$(echo "$card" | json "round(d['card']['winner']['Mumbai Indians'],3)" 2>/dev/null); then
  ok "forecast MI v CSK at Wankhede: MI win chance $w, $(echo "$card" | json "len(d['card']['players'])") batter markets"
else
  bad "forecast: ${card:0:300}"
fi

price=$(get -H "Content-Type: application/json" -d '{"team_a":"Mumbai Indians","team_b":"Chennai Super Kings","venue":"Wankhede Stadium","sims":500,"market":"team_total","team":"Mumbai Indians","line":180.5,"odds_over":1.9,"odds_under":1.9}' "$API_URL/markets/price")
if echo "$price" | json "'verdict' in d and 'price' in d" 2>/dev/null | grep -q True; then
  ok "price check: MI over 180.5 at 1.90 -> fair over $(echo "$price" | json "round(d['price']['fair_over'],2)"), verdict returned"
else
  bad "price check: ${price:0:300}"
fi

code=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/bets")
[ "$code" = 401 ] || [ "$code" = 403 ] && ok "admin routes refuse requests without the token ($code)" || bad "GET /bets without token returned $code"
code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$API_URL/bets")
[ "$code" = 200 ] && ok "admin token accepted, ledger readable" || bad "GET /bets with token returned $code (does DOOSRA_ADMIN_TOKEN match?)"

if [ -n "${SITE_URL:-}" ]; then
  SITE_URL=${SITE_URL%/}
  code=$(curl -s -o /dev/null -w "%{http_code}" "$SITE_URL/")
  [ "$code" = 401 ] && ok "dashboard is password-protected" || bad "dashboard without password returned $code"
  if [ -n "${SITE_PASSWORD:-}" ]; then
    page=$(curl -sS --max-time 120 -u "doosra:$SITE_PASSWORD" "$SITE_URL/forecast?a=Mumbai%20Indians&b=Chennai%20Super%20Kings&venue=Wankhede%20Stadium")
    echo "$page" | grep -q "Chance to win" && ok "dashboard renders a forecast through the API" || bad "dashboard forecast page did not render (check DOOSRA_API_URL on Vercel and DOOSRA_CORS on Cloud Run)"
  fi
fi
exit $fail
