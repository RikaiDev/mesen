#!/usr/bin/env bash
# Industrial QC gate for mesen: lint, format check, tests. Zero new deps.
# Usage: bash scripts/qc.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYBIN=".venv/bin/python"
RUFF=".venv/bin/ruff"
VULTURE=".venv/bin/vulture"
[ -x "$PYBIN" ] || PYBIN="python3"
[ -x "$RUFF" ] || { echo "qc: missing .venv (run: uv sync --extra dev)"; exit 1; }

"$RUFF" check src/ tests/ || exit 1
"$RUFF" format --check src/ tests/ || exit 1
# Dead code at 100% confidence only: the 60% tier is pydantic fields, enum
# members and framework-called methods (FastAPI routes, nn.Module.forward).
# Intentional cases go in vulture_whitelist.py with a consumer note.
if [ -x "$VULTURE" ]; then
	"$VULTURE" src/ vulture_whitelist.py --min-confidence 100 || exit 1
else
	echo "qc: vulture not installed (run: uv sync --extra dev); skipping dead-code gate"
fi
"$PYBIN" -m pytest tests/ -q || exit 1
echo "qc: PASS"
