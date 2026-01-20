# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`btcat-img-dither` is a Python-based image dithering tool that applies monochrome dithering using Austrian flag red (#ED2939). The tool supports advanced shape-based dithering with rectangles, circles, and traditional cut modes, using the Floyd-Steinberg algorithm with optional randomization for organic, natural-looking results.

## Project Structure

- **main.py** - Core Python module containing the dithering logic
- **dither.sh** - Shell wrapper that ensures dependencies are installed and runs the tool
- **pyproject.toml** - UV project configuration with dependencies (Pillow, NumPy, Click)

## Development Setup

This project uses **UV** for Python dependency management and virtual environment handling.

### Prerequisites
- Python 3.10 or higher
- UV package manager

### Install Dependencies
```bash
uv sync
```

UV will automatically create a virtual environment and install all required dependencies.

## Running the Tool

### Basic Usage
```bash
# Default: vertical cut at golden ratio
./dither.sh image.jpg
./dither.sh image.png
```

### Shape-Based Dithering

#### Rectangle Mode
Define one or more rectangular regions. Coordinates: `x1,y1,x2,y2` (fractions, can exceed 0-1 range).

```bash
# Single rectangle: dither right half
./dither.sh --rect=0.5,0,1,1 image.jpg

# Two vertical strips on edges
./dither.sh --rect=0,0,0.1,1 --rect=0.9,0,1,1 image.jpg

# Top and bottom strips
./dither.sh --rect=0,0,1,0.1 --rect=0,0.9,1,1 image.jpg
```

#### Circle Mode
Define one or more circular regions. Coordinates: `x,y,radius` (fractions).

```bash
# Circle in center
./dither.sh --circle=0.5,0.5,0.3 image.jpg

# Multiple circles
./dither.sh --circle=0.25,0.25,0.15 --circle=0.75,0.75,0.15 image.jpg
```

#### Mix Rectangles and Circles
```bash
# Top strip + center circle
./dither.sh --rect=0,0,1,0.1 --circle=0.5,0.5,0.2 image.jpg

# Frame effect with center circle
./dither.sh --rect=0,0,0.05,1 --rect=0.95,0,1,1 --rect=0,0,1,0.05 --rect=0,0.95,1,1 --circle=0.5,0.5,0.25 image.jpg
```

### Global Options

#### Grayscale
Converts the **entire image** to grayscale before applying dithering.

```bash
./dither.sh --grayscale image.jpg
./dither.sh --grayscale --circle=0.5,0.5,0.3 image.jpg
```

**Important**: Grayscale applies to the full image first, then dithering is layered on top. This differs from the original behavior where grayscale only affected non-dithered areas.

#### Fade/Density Control
Control dithering density across all dithered areas (0.0 to 1.0).

```bash
# Sparse: only 10% of pixels dithered
./dither.sh --fade=0.1 image.jpg

# 50% density for subtle effect
./dither.sh --fade=0.5 --rect=0.5,0,1,1 image.jpg

# Combine with grayscale
./dither.sh --grayscale --fade=0.3 --circle=0.5,0.5,0.4 image.jpg
```

### Traditional Cut Mode
For backward compatibility, cut mode still works:

```bash
# Vertical cut at golden ratio
./dither.sh --cut=vertical image.jpg

# Horizontal cut at custom position
./dither.sh --cut=horizontal --pos=0.5 image.jpg
```

**Note**: Cut mode is now internally implemented as a rectangle.

### Direct Python Execution
```bash
uv run python main.py image.jpg
uv run python main.py --rect=0.5,0,1,1 --grayscale image.jpg
```

## How It Works

The tool uses a mask-based architecture for maximum flexibility:

1. **Load Image**: Opens and converts the image to RGB format
2. **Grayscale Conversion** (optional): If `--grayscale` is specified, converts the entire image to grayscale
3. **Create Masks**: Generates boolean masks for rectangles and circles
   - Multiple shapes are combined using logical OR
   - Cut mode creates a default rectangle if no shapes specified
4. **Dithering**: Applies Floyd-Steinberg error diffusion dithering to the entire image
5. **Density Control** (optional): If `--fade` is specified, probabilistically skips pixels for sparse effects
6. **Colorization**: Renders dithered pixels in Austrian flag red (#ED2939) on white background
7. **Compositing**: Applies dithered regions to base image using the masks
8. **Output**: Saved as `[original-name]-dither.[ext]` (with auto-incrementing numbers to avoid overwrites)

## Architecture

### Key Components

#### Mask Creation Functions

- **create_rectangle_mask()** (main.py:37) - Creates boolean mask for rectangular region. Takes `(width, height, x1, y1, x2, y2)` where coordinates are fractions that can be any value. Handles coordinate ordering and clipping to image bounds automatically.

- **create_circle_mask()** (main.py:86) - Creates boolean mask for circular region. Takes `(width, height, center_x, center_y, radius)`. Uses Euclidean distance calculation with NumPy ogrid for efficiency.

- **create_gradient_mask()** (main.py:125) - Creates float density mask for gradient fade effects. Currently used internally but could be exposed for gradient transitions in future versions.

#### Core Dithering Function

- **floyd_steinberg_dither()** (main.py:167) - Implements Floyd-Steinberg error diffusion algorithm with enhancements:
  - **Randomization**: Adds small random noise (±jitter) to threshold on per-pixel basis to break up regular patterns
  - **Density mask**: Accepts optional float mask (0.0 to 1.0) that probabilistically skips pixels for sparse/fade effects
  - **Threshold offset**: Bias parameter for darkness control
  - Error distribution weights: 7/16 right, 3/16 bottom-left, 5/16 bottom, 1/16 bottom-right

#### Utility Functions

- **get_output_filename()** (main.py:245) - Generates output filenames with `-dither` suffix and automatic numbering to prevent overwrites. Preserves the original image format.

- **dither_image()** (main.py:272) - Main processing pipeline. Key parameters:
  - `rectangles`: List of tuples `[(x1, y1, x2, y2), ...]`
  - `circles`: List of tuples `[(cx, cy, r), ...]`
  - `fade`: Density control (0.0 to 1.0)
  - `grayscale_original`: Convert entire image to grayscale first
  - `split_ratio`, `cut_direction`: For backward-compatible cut mode

### Constants

- **AUSTRIAN_RED** = (237, 41, 57) - RGB values for Austrian flag red (#ED2939)
- **GOLDEN_RATIO** = 1.618033988749895 - Used for calculating the default split position

### Coordinate System

All coordinates use fractional values (0.0 to 1.0 representing image dimensions):
- **Rectangles**: `x1,y1,x2,y2` where (x1,y1) is top-left, (x2,y2) is bottom-right
- **Circles**: `x,y,radius` where (x,y) is center
- **Out-of-bounds**: Values outside 0.0-1.0 are accepted and automatically clipped

### Mask Composition

Multiple shapes are combined using `np.logical_or()`, creating a unified boolean mask where any True pixel gets dithered. This allows for complex compositions like frames, patterns, or scattered effects.

## Key Design Decisions & Learnings

### 1. Mask-Based Architecture
**Decision**: Use boolean masks combined with logical OR instead of sequential image cropping/pasting.

**Benefits**:
- Uniform handling of all shape types
- Easy composition of multiple shapes
- Cleaner code with less special-casing
- Foundation for future shape types (ellipses, polygons, etc.)

### 2. Grayscale First, Then Dither
**Decision**: Apply grayscale to entire image before dithering, not just non-dithered areas.

**Rationale**: More intuitive behavior - users expect grayscale to affect the whole image, with dithering as an overlay effect.

**Migration**: Changed from `grayscale_img = img.convert('L').convert('RGB')` applied to split regions, to applying grayscale to full image at load time.

### 3. Uniform Density Control
**Decision**: Make `--fade` apply uniformly to all dithered areas instead of gradient fade for cut mode only.

**Rationale**: Simpler mental model - fade controls "how much" dithering happens everywhere. Gradients can be added as separate feature later if needed.

**Implementation**: `density_mask` is created as uniform value across dithered regions, then passed to Floyd-Steinberg which probabilistically skips pixels.

### 4. Cut Mode as Special Case of Rectangle
**Decision**: Implement cut mode by auto-generating rectangle specs instead of special logic.

**Benefits**:
- Eliminates duplicate code paths
- Maintains backward compatibility
- Reveals that cut mode is just a degenerate case of rectangles

**Implementation**:
```python
if not rectangles and not circles:
    if cut_direction == 'vertical':
        rectangles.append((split_ratio, 0.0, 1.0, 1.0))
    else:
        rectangles.append((0.0, split_ratio, 1.0, 1.0))
```

### 5. Coordinate Flexibility
**Decision**: Allow coordinates outside 0.0-1.0 range instead of validating bounds.

**Benefits**:
- Users can specify partial overlaps naturally (e.g., `-0.1,0,0.1,1` for left edge bleeding off)
- Simpler API (no error handling needed)
- Automatic clipping happens in mask creation

## Dependencies

Managed via UV in pyproject.toml:
- **Pillow** (>=10.0.0) - Image loading, manipulation, and saving
- **NumPy** (>=1.24.0) - Array operations for efficient pixel manipulation and mask operations
- **Click** (>=8.0.0) - CLI framework for argument parsing with automatic validation

## Testing

Test with various modes:

```bash
# Basic functionality
./dither.sh test-image.jpg

# Multiple shapes
./dither.sh --rect=0,0,0.1,1 --rect=0.9,0,1,1 --circle=0.5,0.5,0.2 test-image.jpg

# Grayscale + fade
./dither.sh --grayscale --fade=0.3 --circle=0.5,0.5,0.4 test-image.jpg

# Out-of-bounds coordinates
./dither.sh --rect=-0.1,0,0.1,1 test-image.jpg
```

## Output Behavior

- Output filename: `[original-name]-dither.[ext]`
- If file exists: Appends number (e.g., `image-dither-1.jpg`, `image-dither-2.jpg`)
- Format: Same as input (JPEG quality=95 for JPG, lossless for PNG)
- Never overwrites existing files

## Future Enhancement Ideas

- **Gradient transitions**: Expose `create_gradient_mask()` to CLI for smooth density transitions
- **More shapes**: Ellipses, polygons, custom paths
- **Shape operations**: Subtract, intersect, XOR for complex compositions
- **Per-shape fade**: Allow different density values for different shapes
- **Color variations**: Support multiple dithering colors or patterns
