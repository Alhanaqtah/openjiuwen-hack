#!/usr/bin/env bash
# usage: scripts/ab.sh LABEL TASKS_DIR [ENV=VAL ...]  -> prints score summary; run dir in runs/LABEL
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="$1"; SRC="$2"; shift 2
RUN="$ROOT/runs/$LABEL"; rm -rf "$RUN"; mkdir -p "$RUN"
cp -R "$SRC" "$RUN/tasks"; find "$RUN/tasks" -name answer.txt -delete
set -a; . "$ROOT/.env"; set +a
env "$@" "$ROOT/.venv/bin/python" "$ROOT/run_agent.py" --tasks-dir "$RUN/tasks" --traces "$RUN/traces.jsonl" > "$RUN/log.txt" 2>&1 || true
"$ROOT/.venv/bin/python" "$ROOT/scripts/score.py" "$RUN/tasks" > "$RUN/score.txt"
echo "== $LABEL  $(grep 'done in' "$RUN/log.txt" | sed 's/.*done in/done in/')"
tail -5 "$RUN/score.txt"
