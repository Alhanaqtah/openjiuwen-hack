#!/usr/bin/env bash
# Copy tasks to a scratch dir, run the harness, score against `expected`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$ROOT/tasks}"
RUN="$ROOT/runs/$(date +%m%d-%H%M%S)"
mkdir -p "$RUN"
cp -R "$SRC" "$RUN/tasks"
find "$RUN/tasks" -name answer.txt -delete
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
"$ROOT/.venv/bin/python" "$ROOT/run_agent.py" --tasks-dir "$RUN/tasks" --traces "$RUN/traces.jsonl" 2>&1 | tee "$RUN/log.txt"
"$ROOT/.venv/bin/python" "$ROOT/scripts/score.py" "$RUN/tasks" | tee "$RUN/score.txt"
