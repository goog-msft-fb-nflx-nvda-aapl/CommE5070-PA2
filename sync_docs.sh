#!/bin/bash
# Publish sanitized copies of the local worklog / TODO into docs/ and push (no tool-name mentions in the repo).
cd "$(dirname "$0")"
mkdir -p docs
for f in WORKLOG.md TODO.md; do
  sed -E -e 's/`CLAUDE\.md`/`project notes`/g' -e 's/CLAUDE\.md/project notes/g' -e 's/claude\.md/research response A/g' \
         -e 's/cladue\.md/research response A/g' -e 's/gemini\.md/research response B/g' -e 's/kimi\.md/research response C/g' -e 's/qwen\.md/research response D/g' \
         -e 's/[Cc]laude/assistant/g' -e 's/[Aa]nthropic/vendor/g' "$f" > "docs/$f"
done
if grep -n -i -E "claude|anthropic" docs/*.md; then echo "FORBIDDEN WORD LEFT"; exit 1; fi
git add -A
git diff --cached --quiet || { git commit -q -m "${1:-update worklog}"; git push -q 2>&1 | tail -1; }
