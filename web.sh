#!/usr/bin/env bash
# btcat Dither Explorer — local web UI
# Usage: ./web.sh [port]   (default port: 8050)

set -euo pipefail

PORT="${1:-8050}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Error: 'uv' not found. Install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "Installing / syncing dependencies..."
cd "$SCRIPT_DIR"
uv sync --quiet

echo ""
echo "  btcat Dither Explorer"
echo "  Open in your browser:  http://localhost:${PORT}"
echo "  Press Ctrl+C to stop."
echo ""

exec uv run python web_app.py "$PORT"
