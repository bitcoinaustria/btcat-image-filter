#!/bin/bash
set -e

# Script to generate all example images with consistent seed for reproducibility
# Usage: ./generate-examples.sh [output_dir]
# If output_dir is provided, images are generated there instead of current directory

OUTPUT_DIR="${1:-.}"
SEED=42
INPUT_IMG="test-image-800px.jpg"

echo "Generating example images with seed=$SEED..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Example 1: Basic Grayscale with Cut
echo "Generating example-basic.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pos=0.67 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-basic.jpg"

# Example 2: Multiple Shapes with Grayscale
echo "Generating example-shapes.jpg..."
uv run python main.py "$INPUT_IMG" \
    --rect=0,0,0.2,1 \
    --rect=0.8,0,1,1 \
    --circle=0.5,0.5,0.2 \
    --jitter=33 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-shapes.jpg"

# Example 3: Subtle Dithering with Dark Background
echo "Generating example-fade.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pos=0.5 \
    --fade=0.4 \
    --background=dark \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-fade.jpg"

# Example 4: High Jitter for Organic Texture
echo "Generating example-jitter-high.jpg..."
uv run python main.py "$INPUT_IMG" \
    --circle=0.5,0.5,0.35 \
    --jitter=100 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-jitter-high.jpg"

# Example 6: Glitch Mode
echo "Generating example-glitch.jpg..."
uv run python main.py "$INPUT_IMG" \
    --glitch=0.1 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-glitch.jpg"

# Example 7: Glitch Mode with Fade
echo "Generating example-glitch-fade.jpg..."
uv run python main.py "$INPUT_IMG" \
    --glitch=0.2 \
    --grayscale \
    --fade=0.9 \
    --jitter=10 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-glitch-fade.jpg"

# Example 8: Gradient Density
echo "Generating example-gradient.jpg..."
uv run python main.py "$INPUT_IMG" \
    --circle=0.25,0.5,0.2 \
    --rect=0.5,0,1,1 \
    --gradient=0,1.0,0.1 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-gradient.jpg"

# Fine-tuning: Default
echo "Generating example-default.jpg..."
uv run python main.py "$INPUT_IMG" \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-default.jpg"

# Fine-tuning: High Jitter
echo "Generating example-jitter.jpg..."
uv run python main.py "$INPUT_IMG" \
    --jitter=100 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-jitter.jpg"

# Fine-tuning: Scaled Dots
echo "Generating example-scaled.jpg..."
uv run python main.py "$INPUT_IMG" \
    --reference-width=200 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-scaled.jpg"

# Fine-tuning: Darker
echo "Generating example-dark.jpg..."
uv run python main.py "$INPUT_IMG" \
    --darkness=50 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-dark.jpg"

# Dithering Patterns
echo "Generating example-pattern-ordered.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=ordered \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-ordered.jpg"

echo "Generating example-pattern-atkinson.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=atkinson \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-atkinson.jpg"

echo "Generating example-pattern-clustered.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=clustered-dot \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-clustered.jpg"

echo "Generating example-pattern-bitcoin.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=bitcoin \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-bitcoin.jpg"

echo "Generating example-pattern-hal.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=hal \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-hal.jpg"

# Brand Palettes
echo "Generating example-brand-btcat.jpg..."
uv run python main.py "$INPUT_IMG" \
    --brand=btcat \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-brand-btcat.jpg"

echo "Generating example-brand-lightning.jpg..."
uv run python main.py "$INPUT_IMG" \
    --brand=lightning \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-brand-lightning.jpg"

echo "Generating example-brand-cypherpunk.jpg..."
uv run python main.py "$INPUT_IMG" \
    --brand=cypherpunk \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-brand-cypherpunk.jpg"

echo "Generating example-brand-rgb.jpg..."
uv run python main.py "$INPUT_IMG" \
    --brand=rgb \
    --pattern=ordered \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-brand-rgb.jpg"

echo ""
echo "✓ All example images generated successfully!"
