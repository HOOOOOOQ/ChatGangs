#!/usr/bin/env bash
#
# ChatGangs — one-command setup & run.
#
#   ./quickstart.sh              # setup + train (first run) + start server
#   ./quickstart.sh --no-train   # skip training (rules/lexicon fallback)
#   ./quickstart.sh --test       # setup + train + run the test suite
#
# Tested on Linux/macOS with Python 3.11+. Windows: use WSL or run the
# individual commands listed in docs/INSTALLATION.md.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV=".venv"
TRAIN=1
RUN_TESTS=0

for arg in "$@"; do
  case "$arg" in
    --no-train) TRAIN=0 ;;
    --test) RUN_TESTS=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), f"Python 3.11+ required, found {sys.version.split()[0]}"' \
  || { echo "ChatGangs requires Python 3.11 or newer." >&2; exit 1; }

step "1/4 Virtual environment"
if [[ ! -d "$VENV" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

step "2/4 Dependencies"
if [[ -f requirements-dev.txt ]]; then
  python -m pip install --quiet -r requirements-dev.txt
else
  python -m pip install --quiet -r requirements.txt
fi

step "3/4 Environment file"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example (edit it for production settings)."
else
  echo ".env already exists — left untouched."
fi

if [[ "$TRAIN" == "1" ]]; then
  step "4/4 Training models (real training, ~15 s)"
  python -m scripts.train --task all | tail -n 12
else
  step "4/4 Training skipped (--no-train): using rule/lexicon fallbacks"
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  step "Tests"
  python -m pytest -q
fi

step "Starting ChatGangs"
cat <<'INFO'
Web client : http://localhost:8000/
API docs   : http://localhost:8000/docs
Health     : http://localhost:8000/api/health
Model info : http://localhost:8000/api/model/info

Try it:
  curl -X POST localhost:8000/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"message":"Halo, apa saja kemampuanmu?","session_id":"demo"}'

Press Ctrl+C to stop.
INFO
exec python -m chatgangs --host 0.0.0.0 --port "${PORT:-8000}"
