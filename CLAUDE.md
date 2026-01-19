# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`btcat-img-dither` is a Python-based image dithering tool that applies monochrome dithering to a portion of an image. The tool uses the Floyd-Steinberg dithering algorithm to create a fine pixel-dust effect with Austrian flag red (#ED2939).

## Project Structure

- **main.py** - Core Python module containing the dithering logic
- **dither.sh** - Shell wrapper that ensures dependencies are installed and runs the tool
- **pyproject.toml** - UV project configuration with dependencies (Pillow, NumPy)

## Development Setup

This project uses **UV** for Python dependency management and virtual environment handling.

### Prerequisites
- Python 3.10 or higher
- UV package manager

### Install Dependencies
```bash
uv sync
```

UV will automatically create a virtual environment and install all required dependencies (Pillow, NumPy).

## Running the Tool

### Basic Usage
```bash
# Default: vertical cut at golden ratio with colored original part
./dither.sh image.jpg
./dither.sh image.png
```

### Cut Direction and Position
```bash
# Vertical cut (default): left part original, right part dithered
./dither.sh --cut=vertical image.jpg

# Horizontal cut: top part original, bottom part dithered
./dither.sh --cut=horizontal image.jpg

# Custom position (0.0 to 1.0, default: golden ratio ~0.382)
./dither.sh --cut=vertical --pos=0.4 image.jpg
./dither.sh --cut=horizontal --pos=0.5 image.jpg
```

### Grayscale Option
```bash
# Convert the original (non-dithered) part to grayscale
./dither.sh --grayscale image.jpg

# Combine with other options
./dither.sh --cut=horizontal --pos=0.4 --grayscale image.jpg
```

### Dithering Threshold
```bash
# Adjust threshold (0-255, default: 128)
# Lower = more black, Higher = more white
./dither.sh --threshold 100 image.jpg
```

### Advanced Usage
```bash
# Full customization
./dither.sh --cut=vertical --pos=0.3 --threshold=120 --grayscale image.jpg
```

### Direct Python Execution
```bash
uv run python main.py image.jpg
uv run python main.py --cut=horizontal --pos=0.4 --grayscale image.jpg
```

## How It Works

1. **Load Image**: Opens and converts the image to RGB format
2. **Split**: Cuts the image vertically or horizontally at the specified position (default: golden ratio ~0.382)
3. **Grayscale Original** (optional): Converts the original (non-dithered) part to grayscale if --grayscale is specified
4. **Dithering**: The dithered portion is converted to grayscale and processed using Floyd-Steinberg algorithm
5. **Colorization**: Dithered pixels are rendered in Austrian flag red (#ED2939) on a white background
6. **Recombination**: The original and dithered parts are combined back together
7. **Output**: Saved as `[original-name]-dither.[ext]` (with auto-incrementing numbers to avoid overwrites)

## Architecture

### Key Components

- **floyd_steinberg_dither()** (main.py:35) - Implements the Floyd-Steinberg error diffusion dithering algorithm with optional randomization. Distributes quantization error to neighboring pixels using specific weights (7/16, 3/16, 5/16, 1/16). When randomization is enabled (default), adds small random noise (±15) to the threshold on a per-pixel basis, breaking up regular patterns and creating more organic, natural-looking dithered results while preventing visual artifacts.

- **get_output_filename()** (main.py:57) - Generates output filenames with `-dither` suffix and automatic numbering to prevent overwrites. Preserves the original image format.

- **dither_image()** (main.py:98) - Main processing pipeline that handles image loading, splitting (vertical or horizontal), optional grayscale conversion of original part, dithering, colorization with Austrian red, and saving. Accepts parameters: `split_ratio`, `cut_direction` ('vertical' or 'horizontal'), `threshold`, and `grayscale_original`.

### Constants

- **AUSTRIAN_RED** = (237, 41, 57) - RGB values for Austrian flag red (#ED2939)
- **GOLDEN_RATIO** = 1.618033988749895 - Used for calculating the vertical split position

### Image Splitting

#### Default: Golden Ratio
The split position defaults to `1 / GOLDEN_RATIO ≈ 0.382`, meaning:
- **Vertical cut**: Left ~38.2% original, right ~61.8% dithered
- **Horizontal cut**: Top ~38.2% original, bottom ~61.8% dithered

#### Custom Position
Use `--pos` to specify any split ratio from 0.0 to 1.0:
- `--pos=0.5`: 50/50 split
- `--pos=0.3`: 30% original, 70% dithered
- `--pos=0.7`: 70% original, 30% dithered

#### Grayscale Original
When `--grayscale` is enabled, the original (non-dithered) part is converted to grayscale while the dithered part remains in Austrian red, creating a striking contrast effect.

## Dependencies

Managed via UV in pyproject.toml:
- **Pillow** (>=10.0.0) - Image loading, manipulation, and saving
- **NumPy** (>=1.24.0) - Array operations for efficient pixel manipulation during dithering
- **Click** (>=8.0.0) - CLI framework for argument parsing with automatic validation and better UX

## Testing

To test the tool, you need a sample image:

```bash
# Create a test image or use an existing one
./dither.sh path/to/test-image.jpg

# The output will be saved as: path/to/test-image-dither.jpg
```

## Output Behavior

- Output filename: `[original-name]-dither.[ext]`
- If file exists: Appends number (e.g., `image-dither-1.jpg`, `image-dither-2.jpg`)
- Format: Same as input (JPEG quality=95 for JPG, lossless for PNG)
- Never overwrites existing files
