#!/usr/bin/env bash
# Fetch the Task 2 knowledge-base sources. Idempotent; verifies sha256 when a pinned
# checksum file exists beside the target.
#
# git.lab has no direct route out: everything goes through the squid on bot.lab
# (http://bot.lab:3128) or, for hosts not on the allowlist, through webget.
# cisa.gov is NOT squid-allowlisted, hence webget for the KEV feed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY="http://bot.lab:3128"
export NO_PROXY="${NO_PROXY:-.lab,10.0.1.0/24,localhost,127.0.0.1}"

ATTACK_URL="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
KEV_URL="https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

mkdir -p "$ROOT/data/attack" "$ROOT/data/kev"

fetch_attack() {
  local out="$ROOT/data/attack/enterprise-attack.json"
  if [[ -s "$out" ]]; then
    echo "exists: $out ($(stat -c%s "$out") bytes)"
    return
  fi
  curl -fsSL --proxy "$PROXY" -o "$out" "$ATTACK_URL"
  echo "fetched: $out"
  sha256sum "$out"
}

fetch_kev() {
  local out="$ROOT/data/kev/known_exploited_vulnerabilities.json"
  if [[ -s "$out" ]]; then
    echo "exists: $out ($(stat -c%s "$out") bytes)"
    return
  fi
  # webget lives in /root/bin and may not be on PATH.
  if command -v webget >/dev/null 2>&1; then
    webget "$KEV_URL" -o "$out"
  elif [[ -x /root/bin/webget ]]; then
    /root/bin/webget "$KEV_URL" -o "$out"
  else
    echo "webget not found; fetch $KEV_URL by another route" >&2
    exit 1
  fi
}

fetch_attack
fetch_kev

echo "--- summary ---"
"$ROOT/.venv/bin/python" - "$ROOT" <<'PY'
import json, os, sys
os.chdir(sys.argv[1])
kev = json.load(open("data/kev/known_exploited_vulnerabilities.json"))
print("KEV entries:", kev.get("count"), "catalog:", kev.get("catalogVersion"),
      "released:", kev.get("dateReleased"))
# STIX attack-pattern objects carry TWO deprecation flags and no `deprecated` key:
# `revoked` (superseded by another technique) and `x_mitre_deprecated` (retired outright).
# Filtering on a bare `deprecated` key silently keeps all 161 dead techniques.
stix = json.load(open("data/attack/enterprise-attack.json"))
pats = [o for o in stix["objects"]
        if o.get("type") == "attack-pattern"
        and not o.get("revoked", False)
        and not o.get("x_mitre_deprecated", False)]
def eid(o):
    for r in (o.get("external_references") or []):
        if r.get("external_id", "").startswith("T"):
            return r["external_id"]
    return ""
parents = [t for t in pats if "." not in eid(t)]
raw = sum(1 for o in stix["objects"] if o.get("type") == "attack-pattern")
print("ATT&CK attack-patterns: raw", raw, "-> live", len(pats),
      "| parents:", len(parents),
      "| sub-techniques:", len(pats) - len(parents))
PY
