#!/bin/bash
# Compatibility wrapper; GitHub Actions now owns scheduled publishing.
set -euo pipefail
exec bash "$(dirname "$0")/daily_nhl_refresh.sh" "$@"
