#!/bin/bash
set -e

# Script to generate all example images with consistent seed for reproducibility
# Usage: ./generate-examples.sh [output_dir]
# If output_dir is provided, images are generated there instead of current directory

OUTPUT_DIR="${1:-.}"
SEED=42
INPUT_IMG="examples/test-image-1920px.jpg"

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
    --output="$OUTPUT_DIR/example-basic.webp"

# Example 2: Multiple Shapes with Grayscale
echo "Generating example-shapes.jpg..."
uv run python main.py "$INPUT_IMG" \
    --rect=0,0,0.2,1 \
    --rect=0.8,0,1,1 \
    --circle=0.5,0.5,0.2 \
    --jitter=33 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-shapes.webp"

# Example 3: Subtle Dithering with Dark Background
echo "Generating example-fade.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pos=0.5 \
    --fade=0.4 \
    --background=dark \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-fade.webp"

# Example 4: High Jitter for Organic Texture
echo "Generating example-jitter-high.jpg..."
uv run python main.py "$INPUT_IMG" \
    --circle=0.5,0.5,0.35 \
    --jitter=100 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-jitter-high.webp"

# Example 5: Satoshi Mode
echo "Generating example-satoshi.jpg..."
uv run python main.py "$INPUT_IMG" \
    --circle=0.5,0.5,0.3 \
    --satoshi-mode \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-satoshi.webp"

# Example 6: Glitch Mode
echo "Generating example-glitch.jpg..."
uv run python main.py "$INPUT_IMG" \
    --glitch=0.02 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-glitch.webp"

# Example 7: Glitch Mode with Fade
echo "Generating example-glitch-fade.jpg..."
uv run python main.py "$INPUT_IMG" \
    --glitch=0.1 \
    --grayscale \
    --fade=0.9 \
    --jitter=10 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-glitch-fade.webp"

# Example 8: Gradient Density
echo "Generating example-gradient.jpg..."
uv run python main.py "$INPUT_IMG" \
    --circle=0.25,0.5,0.2 \
    --rect=0.5,0,1,1 \
    --gradient=0,1.0,0.1 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-gradient.webp"

# Example 9: Gradient Density (Reversed)
echo "Generating example-gradient-reverse.jpg..."
uv run python main.py "$INPUT_IMG" \
    --circle=0.25,0.5,0.2 \
    --rect=0.5,0,1,1 \
    --gradient=180,1.0,0.1 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-gradient-reverse.webp"

# Example 10: Shaded Dithering with Quantization
echo "Generating example-shade-quantized.jpg..."
uv run python main.py "$INPUT_IMG" \
    --shade="1,q=4" \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-shade-quantized.webp"

# Example 11: Shaded Dithering with Quantization and Glitch
echo "Generating example-shade-quantized-glitch.jpg..."
uv run python main.py "$INPUT_IMG" \
    --shade="1,q=4" \
    --glitch=0.02 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-shade-quantized-glitch.webp"

# Fine-tuning: Default
echo "Generating example-default.jpg..."
uv run python main.py "$INPUT_IMG" \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-default.webp"

# Fine-tuning: High Jitter
echo "Generating example-jitter.jpg..."
uv run python main.py "$INPUT_IMG" \
    --jitter=100 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-jitter.webp"

# Fine-tuning: Scaled Dots
echo "Generating example-scaled.jpg..."
uv run python main.py "$INPUT_IMG" \
    --reference-width=200 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-scaled.webp"

# Fine-tuning: Darker
echo "Generating example-dark.jpg..."
uv run python main.py "$INPUT_IMG" \
    --darkness=50 \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-dark.webp"

# Dithering Patterns
echo "Generating example-pattern-ordered.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=ordered \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-ordered.webp"

echo "Generating example-pattern-atkinson.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=atkinson \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-atkinson.webp"

echo "Generating example-pattern-clustered.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=clustered-dot \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-clustered.webp"

echo "Generating example-pattern-bitcoin.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=bitcoin \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-bitcoin.webp"

echo "Generating example-pattern-hal.jpg..."
uv run python main.py "$INPUT_IMG" \
    --pattern=hal \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-hal.webp"

echo "Generating example-pattern-blue-noise.webp..."
uv run python main.py "$INPUT_IMG" \
    --pattern=blue-noise \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-pattern-blue-noise.webp"

# Brand Palettes
echo "Generating example-brand-btcat.jpg..."
uv run python main.py "$INPUT_IMG" \
    --brand=btcat \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-brand-btcat.webp"

echo "Generating example-brand-lightning.jpg..."
uv run python main.py "$INPUT_IMG" \
    --brand=lightning \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-brand-lightning.webp"

echo "Generating example-brand-cypherpunk.jpg..."
uv run python main.py "$INPUT_IMG" \
    --brand=cypherpunk \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-brand-cypherpunk.webp"

echo "Generating example-brand-rgb.jpg..."
uv run python main.py "$INPUT_IMG" \
    --brand=rgb \
    --pattern=ordered \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-brand-rgb.webp"

# Example 12: Original Mode (nbb)
echo "Generating example-original.jpg..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original.webp"

# Example 13: Original Mode with Glitch
echo "Generating example-original-glitch.jpg..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --glitch=0.02 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original-glitch.webp"

# Example 14: Original Mode with circle mask
echo "Generating example-original-circle.jpg..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --circle=0.5,0.5,0.3 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original-circle.webp"

# Example 15: Original Mode with fade
echo "Generating example-original-fade.jpg..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --fade=0.5 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original-fade.webp"

# Example 16: Original Mode with gradient
echo "Generating example-original-gradient.jpg..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --gradient=0,0.1,1.0 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original-gradient.webp"

# Example 17: Original Mode without bloom
echo "Generating example-original-nobloom.jpg..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --bloom-intensity=0 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original-nobloom.webp"

# Example 18: Original Mode with brightness/contrast
echo "Generating example-original-bright.jpg..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --brightness=1.3 \
    --contrast=1.2 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original-bright.webp"

# Example 19: Original Mode with point-size=4
echo "Generating example-original-ps4.webp..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --point-size=4 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original-ps4.webp"

# Example 20: Original Mode with point-size=2 and detail reduction
echo "Generating example-original-ps2.jpg..."
uv run python main.py "$INPUT_IMG" \
    --mode=original \
    --grayscale \
    --detail=0.7 \
    --point-size=2 \
    --seed=$SEED \
    --output="$OUTPUT_DIR/example-original-ps2.webp"

echo ""
echo "✓ All example images generated successfully!"