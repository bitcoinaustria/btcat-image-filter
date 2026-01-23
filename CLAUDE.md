# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`btcat-img-dither` is a Python-based image dithering tool that applies monochrome dithering using Austrian flag red (#E3000F). The tool supports advanced shape-based dithering with rectangles, circles, and traditional cut modes, using the Floyd-Steinberg algorithm with optional randomization for organic, natural-looking results.

More info is in ./README.md

**Note:** Always use `--grayscale` when generating example images unless specifically instructed otherwise. This ensures a consistent and high-quality aesthetic.

**Note:** Always use `--grayscale` when generating example images unless specifically instructed otherwise. This ensures a consistent and high-quality aesthetic.

## Project Structure

- **main.py** - Wrapper script for backward compatibility
- **dither.sh** - Shell wrapper that ensures dependencies are installed and runs the tool
- **btcat_images/** - Main package source code
  - **cli.py** - Command-line interface
  - **core/** - Core logic (pipeline, utils)
  - **processing/** - Image processing algorithms (dither, filters, masks)
  - **tui/** - Terminal User Interface
  - **constants.py** - Global constants
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
./effect.sh image.jpg
./effect.sh image.png
```

### Shape-Based Dithering

#### Rectangle Mode
Define one or more rectangular regions. Coordinates: `x1,y1,x2,y2` (fractions, can exceed 0-1 range).

```bash
# Single rectangle: dither right half
./effect.sh --rect=0.5,0,1,1 image.jpg

# Two vertical strips on edges
./effect.sh --rect=0,0,0.1,1 --rect=0.9,0,1,1 image.jpg

# Top and bottom strips
./effect.sh --rect=0,0,1,0.1 --rect=0,0.9,1,1 image.jpg
```

#### Circle Mode
Define one or more circular regions. Coordinates: `x,y,radius` (fractions).

```bash
# Circle in center
./effect.sh --circle=0.5,0.5,0.3 image.jpg

# Multiple circles
./effect.sh --circle=0.25,0.25,0.15 --circle=0.75,0.75,0.15 image.jpg
```

#### Mix Rectangles and Circles
```bash
# Top strip + center circle
./effect.sh --rect=0,0,1,0.1 --circle=0.5,0.5,0.2 image.jpg

# Frame effect with center circle
./effect.sh --rect=0,0,0.05,1 --rect=0.95,0,1,1 --rect=0,0,1,0.05 --rect=0,0.95,1,1 --circle=0.5,0.5,0.25 image.jpg
```

### Global Options

#### Grayscale
Converts the **entire image** to grayscale before applying dithering.

```bash
./effect.sh --grayscale image.jpg
./effect.sh --grayscale --circle=0.5,0.5,0.3 image.jpg
```

**Important**: Grayscale applies to the full image first, then dithering is layered on top. This differs from the original behavior where grayscale only affected non-dithered areas.

#### Fade/Density Control
Control dithering density across all dithered areas (0.0 to 1.0).

```bash
# Sparse: only 10% of pixels dithered
./effect.sh --fade=0.1 image.jpg

# 50% density for subtle effect
./effect.sh --fade=0.5 --rect=0.5,0,1,1 image.jpg

# Combine with grayscale
./effect.sh --grayscale --fade=0.3 --circle=0.5,0.5,0.4 image.jpg
```

#### Gradient Density
Create gradual density transitions across the image using angle-based gradients.

Format: `--gradient=angle,start,end`
- **Angle**: 0-360° (0=left→right, 90=top→bottom, 180=right→left, 270=bottom→top)
- **Start**: Density at gradient start (0.0 to 1.0)
- **End**: Density at gradient end (0.0 to 1.0)

```bash
# Horizontal gradient: sparse on left, dense on right
./effect.sh --gradient=0,0.1,1.0 image.jpg

# Vertical gradient: fade from top to bottom
./effect.sh --gradient=90,0.0,1.0 image.jpg

# Reverse horizontal: dense on left, sparse on right
./effect.sh --gradient=180,1.0,0.2 image.jpg

# Diagonal gradient (45 degrees)
./effect.sh --gradient=45,0.0,1.0 image.jpg

# Combine with shapes: gradient only in specific areas
./effect.sh --gradient=0,0.1,1.0 --rect=0.3,0,0.7,1 image.jpg

# Create fade-out effect: image gradually vanishes to white
./effect.sh --gradient=0,1.0,0.0 image.jpg
```

**Note**: Gradient overrides `--fade` if both are specified. Gradient is computed based on pixel position in the image, then applied only to dithered areas defined by masks.

### Traditional Cut Mode
For backward compatibility, cut mode still works:

```bash
# Vertical cut at golden ratio
./effect.sh --cut=vertical image.jpg

# Horizontal cut at custom position
./effect.sh --cut=horizontal --pos=0.5 image.jpg
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
5. **Density Control** (optional):
   - If `--gradient` is specified, creates angle-based gradient density mask across entire image
   - If `--fade` is specified (and no gradient), creates uniform density mask
   - Density mask probabilistically skips pixels for sparse/fade effects
6. **Colorization**: Renders dithered pixels in Austrian flag red (#E3000F) on white background
7. **Compositing**: Applies dithered regions to base image using the masks
8. **Output**: Saved as `[original-name]-dither.[ext]` (with auto-incrementing numbers to avoid overwrites)

## Architecture

### Key Components

#### Mask Creation Functions

- **create_rectangle_mask()** (main.py:86) - Creates boolean mask for rectangular region. Takes `(width, height, x1, y1, x2, y2)` where coordinates are fractions that can be any value. Handles coordinate ordering and clipping to image bounds automatically.

- **create_circle_mask()** (main.py:135) - Creates boolean mask for circular region. Takes `(width, height, center_x, center_y, radius)`. Uses Euclidean distance calculation with NumPy ogrid for efficiency.

- **create_gradient_mask()** (main.py:174) - Legacy function for cut-mode gradient fade effects. Creates gradient from split line to edge.

- **create_gradient_density_mask()** (main.py:216) - Creates angle-based gradient density mask for general gradient transitions. Takes `(width, height, angle, density_start, density_end)`:
  - **Angle**: 0-360° determines gradient direction using trigonometric projection
  - Projects each pixel position onto gradient direction vector
  - Normalizes projection to [0, 1] range
  - Maps to [density_start, density_end] for final density values
  - Supports arbitrary angles for diagonal, circular, or custom gradient patterns

#### Core Dithering Function

- **floyd_steinberg_dither()** (main.py:167) - Implements Floyd-Steinberg error diffusion algorithm with enhancements:
  - **Randomization**: Adds small random noise (±jitter) to threshold on per-pixel basis to break up regular patterns
  - **Density mask**: Accepts optional float mask (0.0 to 1.0) that probabilistically skips pixels for sparse/fade effects
  - **Threshold offset**: Bias parameter for darkness control
  - Error distribution weights: 7/16 right, 3/16 bottom-left, 5/16 bottom, 1/16 bottom-right

#### Utility Functions

- **get_output_filename()** (main.py:245) - Generates output filenames with `-dither` suffix and automatic numbering to prevent overwrites. Preserves the original image format.

- **dither_image()** (main.py:928) - Main processing pipeline. Key parameters:
  - `rectangles`: List of tuples `[(x1, y1, x2, y2), ...]`
  - `circles`: List of tuples `[(cx, cy, r), ...]`
  - `fade`: Uniform density control (0.0 to 1.0)
  - `gradient`: Tuple `(angle, density_start, density_end)` for angle-based gradient
  - `grayscale_original`: Convert entire image to grayscale first
  - `split_ratio`, `cut_direction`: For backward-compatible cut mode

### Constants

- **AUSTRIAN_RED** = (227, 0, 15) - RGB values for Austrian flag red (#E3000F)
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

### 3. Uniform Density Control vs Gradients
**Decision**: Separate `--fade` (uniform) from `--gradient` (angle-based) for clear mental model.

**Rationale**:
- `--fade` provides simple uniform density control across all dithered areas
- `--gradient` enables complex gradual transitions with full directional control
- Gradient overrides fade when both specified

**Implementation**:
- Uniform: `density_mask = np.full((h, w), fade)`
- Gradient: `density_mask = create_gradient_density_mask(w, h, angle, start, end)`
- Both masks passed to dithering algorithm which probabilistically skips pixels

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

### 6. Angle-Based Gradient System
**Decision**: Use angle (0-360°) + projection instead of enum-based directions.

**Benefits**:
- Supports arbitrary gradient directions (diagonal, any angle)
- Intuitive: 0°=horizontal, 90°=vertical, 45°=diagonal
- Single API for all gradient types
- Mathematical elegance: gradient = dot product of position with direction vector

**Implementation**:
```python
# Convert angle to direction vector
dx = np.cos(np.deg2rad(angle))
dy = np.sin(np.deg2rad(angle))

# Project each pixel position onto gradient direction
projection = x_norm * dx + y_norm * dy

# Normalize to [0, 1] and map to [density_start, density_end]
```

**Alternative Considered**: Enum-based directions ('horizontal', 'vertical', 'diagonal-nw', etc.) - rejected as too limiting and verbose for arbitrary angles.

## Dependencies

Managed via UV in pyproject.toml:
- **Pillow** (>=10.0.0) - Image loading, manipulation, and saving
- **NumPy** (>=1.24.0) - Array operations for efficient pixel manipulation and mask operations
- **Click** (>=8.0.0) - CLI framework for argument parsing with automatic validation

## Testing

Test with various modes:

```bash
# Basic functionality
./effect.sh test-image.jpg

# Multiple shapes
./effect.sh --rect=0,0,0.1,1 --rect=0.9,0,1,1 --circle=0.5,0.5,0.2 test-image.jpg

# Grayscale + fade
./effect.sh --grayscale --fade=0.3 --circle=0.5,0.5,0.4 test-image.jpg

# Gradient transitions
./effect.sh --gradient=0,0.1,1.0 test-image.jpg
./effect.sh --gradient=90,0.0,1.0 --rect=0.3,0,0.7,1 test-image.jpg

# Out-of-bounds coordinates
./effect.sh --rect=-0.1,0,0.1,1 test-image.jpg
```

## Output Behavior

- Output filename: `[original-name]-dither.[ext]`
- If file exists: Appends number (e.g., `image-dither-1.jpg`, `image-dither-2.jpg`)
- Format: Same as input (JPEG quality=95 for JPG, lossless for PNG)
- Never overwrites existing files

## Future Enhancement Ideas

- **More shapes**: Ellipses, polygons, custom paths
- **Shape operations**: Subtract, intersect, XOR for complex compositions
- **Per-shape density**: Allow different gradient/fade values for different shapes
- **Radial gradients**: Distance-based gradients from a center point
- **Color variations**: Support multiple dithering colors or patterns
- **Custom gradient curves**: Non-linear gradient mapping (ease-in, ease-out, etc.)