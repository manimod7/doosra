#!/usr/bin/env bash
# Fails if anything that looks like a real secret is in the tracked files.
# Runs in CI on every push; install locally with:  ln -s ../../scripts/check-secrets.sh .git/hooks/pre-commit
# Secrets belong in Cloud Run variables, Vercel env vars and GitHub Actions secrets, never in code.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

patterns=(
  'npg_[A-Za-z0-9]{8,}'                                   # Neon passwords
  '[a-z0-9-]+\.[a-z0-9-]+\.aws\.neon\.tech'               # Neon hosts
  'postgres(ql)?://[^:@/[:space:]]+:[^@[:space:]]{6,}@[^/[:space:]]*\.[a-z]{2,}' # DB URLs with a password and a real host
  'sk-ant-[A-Za-z0-9_-]{20,}'                             # Anthropic
  'AIza[0-9A-Za-z_-]{30,}'                                # Google / Gemini
  'gsk_[A-Za-z0-9]{20,}'                                  # Groq
  'gh[pousr]_[A-Za-z0-9]{30,}'                            # GitHub tokens
  '(api[_-]?key|token|secret|password)["'"'"' ]*[:=][ "'"'"']*[A-Za-z0-9_-]{24,}' # key = long literal
  '\b[a-f0-9]{32}\b'                                      # 32-hex keys (The Odds API style)
)
# files that legitimately hold look-alikes: lockfiles (integrity hashes) and this script
exclude=(':!*package-lock.json' ':!scripts/check-secrets.sh')

found=0
for p in "${patterns[@]}"; do
  if hits=$(git grep -I -n -E -i "$p" -- . "${exclude[@]}"); then
    echo "$hits" | sed -E 's/(:[0-9]+:).*/\1 matches a secret pattern/'
    found=1
  fi
done
if [ "$found" = 1 ]; then
  echo "Possible secret in tracked files (values not shown). Move it to an environment variable." >&2
  exit 1
fi
echo "No secrets found in tracked files."
