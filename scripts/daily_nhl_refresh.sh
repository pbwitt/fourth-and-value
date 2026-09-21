#!/bin/bash
# Build current regular-season pages. Publishing is handled by NHL Daily Update.
set -euo pipefail
cd "$(dirname "$0")/.."
NHL_PYTHON="${PY:-.venv/bin/python}"
exec "$NHL_PYTHON" scripts/nhl/refresh.py "$@"
