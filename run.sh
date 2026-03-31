#!/usr/bin/env bash
set -euo pipefail
# Always use this project's uv environment (.venv next to pyproject.toml).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$#" -eq 0 ]; then
  exec uv run app.py --point-tracker-port 7890 --track-server "tcp://192.168.31.50:5555"
else
  exec uv run app.py "$@"
fi
