#!/usr/bin/env bash
# Industrial QC gate for mesen: lint, format check, tests. Zero new deps.
# Usage: bash scripts/qc.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYBIN=".venv/bin/python"
RUFF=".venv/bin/ruff"
[ -x "$PYBIN" ] || PYBIN="python3"
[ -x "$RUFF" ] || { echo "qc: missing .venv (run: uv sync --extra dev)"; exit 1; }

"$RUFF" check src/ tests/ || exit 1
"$RUFF" format --check src/ tests/ || exit 1
"$PYBIN" -m pytest tests/ -q || exit 1
echo "qc: PASS"
