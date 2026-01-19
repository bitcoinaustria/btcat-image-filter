# btcat-img-dither

Image dithering tool that applies monochrome dithering to a portion of an image using Austrian flag red.

## Features

- **Flexible image splitting**: Vertical or horizontal cuts at any position
- **Golden ratio default**: Splits images at the golden ratio (~38%/~62%) by default
- **Randomized dithering**: Floyd-Steinberg with threshold randomization for organic, less regular patterns
- **Austrian flag red**: Uses #ED2939 for dithered pixels
- **Grayscale option**: Optionally convert the original (non-dithered) part to grayscale
- **Format preservation**: Maintains original image format (JPEG/PNG)
- **Smart naming**: Automatic output naming with collision avoidance
- **Zero-configuration**: Ready to run out of the box with UV

## Example

### Input Image
![Original Image](test-image-800px.jpg)

### Command
```bash
./dither.sh --grayscale test-image-800px.jpg
```

### Output Image
![Dithered Image](test-image-800px-dither.jpg)

The left ~38% is converted to grayscale, and the right ~62% is dithered in Austrian flag red (#ED2939).

### Configuration Examples

| Option | Description | Example Output |
|--------|-------------|----------------|
| **Default** | Standard settings | ![Default](example-default.jpg) |
| **High Jitter** | `... --jitter 100` <br> Increases randomness/noise | ![Jitter](example-jitter.jpg) |
| **Scaled Dots** | `... --reference-width 200` <br> Makes dots larger (proportional to width) | ![Scaled](example-scaled.jpg) |
| **Darker** | `... --darkness 50` <br> Draws fewer background pixels (darker appearance) | ![Darker](example-dark.jpg) |


## Installation

This project uses UV for dependency management. If you don't have UV installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install dependencies:

```bash
uv sync
```

## Usage

### Basic Usage

```bash
# Default: vertical cut at golden ratio
./dither.sh image.jpg

# Works with PNG too
./dither.sh image.png
```

### Cut Direction and Position

```bash
# Vertical cut (default): left part original, right part dithered
./dither.sh --cut=vertical image.jpg

# Horizontal cut: top part original, bottom part dithered
./dither.sh --cut=horizontal image.jpg

# Custom position (0.0 to 1.0)
# For vertical: --pos=0.4 means left 40% original, right 60% dithered
./dither.sh --cut=vertical --pos=0.4 image.jpg

# For horizontal: --pos=0.4 means top 40% original, bottom 60% dithered
./dither.sh --cut=horizontal --pos=0.4 image.jpg
```

### Grayscale Option

```bash
# Convert the original (non-dithered) part to grayscale
./dither.sh --grayscale image.jpg

# Combine with other options
./dither.sh --cut=horizontal --pos=0.5 --grayscale image.jpg
```

### Dithering Threshold

```bash
# Adjust the dithering threshold (0-255, default: 128)
# Lower values = more black pixels, Higher values = more white pixels
./dither.sh --threshold 100 image.jpg
```

### Randomization & Jitter

```bash
# Randomization is enabled by default. Control the amount with --jitter (default: 30.0)
./dither.sh --jitter 50 image.jpg

# Disable randomization for classic Floyd-Steinberg
./dither.sh --no-randomize image.jpg
```

### Point Size Scaling

Make the dither pattern proportional to the image width. This ensures images of different resolutions have similar visual texture density.

```bash
# Set a reference width (default: 1024).
# If input is wider than this, dither points become larger.
./dither.sh --reference-width 800 image.jpg
```

### Density / Darkness Control

Adjust the density of the dithered output.

```bash
# Make the result darker (fewer white pixels)
./dither.sh --darkness 30 image.jpg

# Make the result lighter (more white pixels)
./dither.sh --darkness -30 image.jpg
```

### Advanced Examples

```bash
# Full customization
./dither.sh --cut=vertical --pos=0.3 --threshold=120 --grayscale image.jpg

# 50/50 horizontal split with grayscale top
./dither.sh --cut=horizontal --pos=0.5 --grayscale image.jpg
```

## Output

The tool generates output files with the format:
- `[original-name]-dither.[ext]`
- If the file exists, it appends a number: `[original-name]-dither-1.[ext]`

## How It Works

1. **Load and prepare**: Reads the input image and converts to RGB if needed
2. **Split the image**: Cuts the image vertically or horizontally at the specified position (default: golden ratio)
3. **Grayscale conversion** (optional): Converts the original (non-dithered) part to grayscale
4. **Dithering**: Applies Floyd-Steinberg error diffusion dithering to the dithered portion
5. **Colorization**: Renders dithered pixels in Austrian flag red (#ED2939) on white background
6. **Recombination**: Combines the original and dithered parts back together
7. **Save**: Outputs the result in the same format as the input with smart naming

## Requirements

- Python 3.10+
- Pillow (for image processing)
- NumPy (for array operations)
- Click (for CLI with automatic validation)

All dependencies are managed automatically by UV.

## License

Apache License 2.0

Copyright 2026 Harald Schilly <hsy@bitcoin-austria.at>

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
