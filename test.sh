#!/bin/bash
set -e

echo "Installing/Syncing dependencies..."
uv sync --quiet

echo "Running pyright..."
uv run pyright main.py

echo "Running unit tests..."
uv run pytest

echo "Running e2e tests..."
echo ""

# Create temporary directory for generated images
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Generating all example images in temporary directory..."
./generate-examples.sh "$TEMP_DIR" > /dev/null 2>&1

# List of all example images to test
EXAMPLES=(
    "example-basic.webp"
    "example-shapes.webp"
    "example-fade.webp"
    "example-jitter-high.webp"
    "example-satoshi.webp"
    "example-glitch.webp"
    "example-glitch-fade.webp"
    "example-default.webp"
    "example-jitter.webp"
    "example-scaled.webp"
    "example-dark.webp"
    "example-pattern-ordered.webp"
    "example-pattern-atkinson.webp"
    "example-pattern-clustered.webp"
    "example-pattern-bitcoin.webp"
    "example-pattern-hal.webp"
    "example-pattern-blue-noise.webp"
    "example-brand-btcat.webp"
    "example-brand-lightning.webp"
    "example-brand-cypherpunk.webp"
    "example-brand-rgb.webp"
    "example-shade-quantized.webp"
    "example-shade-quantized-glitch.webp"
    "example-original.webp"
    "example-original-glitch.webp"
    "example-original-circle.webp"
    "example-original-fade.webp"
    "example-original-gradient.webp"
    "example-original-nobloom.webp"
    "example-original-bright.webp"
    "example-original-ps4.webp"
    "example-original-ps2.webp"
)

# Compare each generated image with reference
PASSED=0
FAILED=0
MISSING=0

for example in "${EXAMPLES[@]}"; do
    if [ ! -f "examples/$example" ]; then
        echo "⚠️  MISSING: examples/$example (reference not in repo)"
        MISSING=$((MISSING + 1))
        continue
    fi

    if cmp -s "$TEMP_DIR/$example" "examples/$example"; then
        echo "✓ PASSED: $example"
        PASSED=$((PASSED + 1))
    else
        echo "✗ FAILED: $example (output differs from reference)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test Results:"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Missing: $MISSING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "❌ Some tests failed!"
    echo "Generated images are in: $TEMP_DIR"
    echo "Compare with: cmp <generated> <reference>"
    trap - EXIT  # Don't delete temp dir on failure
    exit 1
fi

if [ $MISSING -gt 0 ]; then
    echo "⚠️  Some reference images are missing (not committed to repo)"
    echo "Run: ./generate-examples.sh examples/ && git add -f examples/example-*.webp"
fi

echo "✓ All e2e tests passed!"