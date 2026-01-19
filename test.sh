#!/bin/bash
set -e

echo "Installing/Syncing dependencies..."
uv sync --quiet

echo "Running pyright..."
uv run pyright main.py

echo "Running pytype..."
uv run pytype main.py

echo "Running e2e test..."
# Generate a dithered image with a fixed seed
INPUT_IMG="test-image-800px.jpg"
# Output as PNG for stable comparison
OUTPUT_IMG="test-image-800px-dither.png"
REFERENCE_IMG="reference-dither.png"
SEED=42

# Clean up previous run
rm -f "$OUTPUT_IMG"

# Run the tool
# Using --output to force PNG format
uv run python main.py "$INPUT_IMG" --seed "$SEED" --output "$OUTPUT_IMG"

# Check if reference image exists
if [ ! -f "$REFERENCE_IMG" ]; then
    echo "Warning: Reference image $REFERENCE_IMG not found."
    echo "Creating it now from the current output to serve as a baseline."
    cp "$OUTPUT_IMG" "$REFERENCE_IMG"
    echo "Reference image created. Please verify it visually."
else
    # Compare output with reference
    if cmp -s "$OUTPUT_IMG" "$REFERENCE_IMG"; then
        echo "e2e test passed: Output matches reference."
    else
        echo "e2e test failed: Output differs from reference."
        echo "Expected: $REFERENCE_IMG"
        echo "Got: $OUTPUT_IMG"
        exit 1
    fi
fi

# Clean up
rm -f "$OUTPUT_IMG"

echo "All tests passed!"
