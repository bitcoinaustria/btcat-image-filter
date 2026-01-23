#!/usr/bin/env bash
# Copyright 2026 Harald Schilly <hsy@bitcoin-austria.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shell wrapper for btcat-img-dither tool
# Ensures dependencies are installed and runs the Python dithering script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# Sync dependencies if needed (uv will handle the virtual environment)
cd "$SCRIPT_DIR"
uv sync --quiet 2>/dev/null || true

# Run the Python script with all arguments passed through
uv run python main.py "$@"
