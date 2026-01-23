#!/usr/bin/env python3
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

"""
Image dithering tool that applies monochrome dithering to a portion of an image.
Uses Austrian flag red (#ED2939) and applies dithering to the right side after a golden ratio cut.
"""

import sys
import click
from pathlib import Path
from PIL import Image
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Tuple, Literal
from numba import jit


# Brand definitions
BRANDS = {
    'btcat': {'color': (227, 0, 15), 'type': 'monochrome'},       # #E3000F
    'lightning': {'color': (245, 155, 31), 'type': 'monochrome'}, # #F59B1F
    'cypherpunk': {'color': (0, 255, 65), 'type': 'monochrome'},  # #00FF41
    'rgb': {'type': 'rgb'}
}

# Dark background color for dithered areas
DARK_BACKGROUND: Tuple[int, int, int] = (34, 34, 34)  # #222222

# Golden ratio (phi)
GOLDEN_RATIO: float = 1.618033988749895

# Dithering patterns
DitherPattern = Literal['floyd-steinberg', 'ordered', 'atkinson', 'clustered-dot', 'bitcoin', 'hal']

# Matrices
# Bayer 8x8 matrix
BAYER_8x8 = np.array([
    [ 0, 48, 12, 60,  3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [ 8, 56,  4, 52, 11, 59,  7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [ 2, 50, 14, 62,  1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58,  6, 54,  9, 57,  5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21]
], dtype=float) / 64.0

# Clustered dot (Spiral 8x8)
CLUSTERED_8x8 = np.array([
    [24, 10, 12, 26, 35, 47, 49, 37],
    [ 8,  0,  2, 14, 45, 59, 61, 51],
    [22,  6,  4, 16, 43, 57, 63, 53],
    [30, 20, 18, 28, 33, 41, 55, 39],
    [36, 50, 48, 34, 27, 13, 11, 25],
    [52, 62, 60, 46, 15,  3,  1,  9],
    [54, 64, 58, 44, 17,  5,  7, 23],
    [40, 56, 42, 32, 29, 19, 21, 31]
], dtype=float) / 65.0

# Bitcoin custom pattern (8x8)
# Creates a "B" shape for Bitcoin branding with two humps
BITCOIN_8x8 = np.array([
    [10, 60, 60, 60, 60, 10, 10, 10],
    [10, 60, 10, 10, 10, 60, 10, 10],
    [10, 60, 10, 10, 10, 60, 10, 10],
    [10, 60, 60, 60, 60, 10, 10, 10],
    [10, 60, 10, 10, 10, 60, 10, 10],
    [10, 60, 10, 10, 10, 60, 10, 10],
    [10, 60, 60, 60, 60, 10, 10, 10],
    [10, 10, 10, 10, 10, 10, 10, 10],
], dtype=float) / 64.0


def create_rectangle_mask(
    width: int,
    height: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> npt.NDArray[np.bool_]:
    """
    Create a rectangular mask for dithering.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        x1: Left X coordinate (fraction, can be any value)
        y1: Top Y coordinate (fraction, can be any value)
        x2: Right X coordinate (fraction, can be any value)
        y2: Bottom Y coordinate (fraction, can be any value)

    Returns:
        Boolean mask array where True indicates dithering area
    """
    mask = np.zeros((height, width), dtype=bool)

    # Convert fractions to pixel coordinates
    px1 = int(x1 * width)
    py1 = int(y1 * height)
    px2 = int(x2 * width)
    py2 = int(y2 * height)

    # Ensure coordinates are in correct order
    if px1 > px2:
        px1, px2 = px2, px1
    if py1 > py2:
        py1, py2 = py2, py1

    # Clip to image bounds
    px1 = max(0, min(px1, width))
    px2 = max(0, min(px2, width))
    py1 = max(0, min(py1, height))
    py2 = max(0, min(py2, height))

    # Fill the rectangle
    if px2 > px1 and py2 > py1:
        mask[py1:py2, px1:px2] = True

    return mask


def create_circle_mask(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float
) -> npt.NDArray[np.bool_]:
    """
    Create a circular mask for dithering.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        center_x: Center X coordinate (0.0 to 1.0)
        center_y: Center Y coordinate (0.0 to 1.0)
        radius: Radius (0.0 to 1.0, fraction of image dimensions)

    Returns:
        Boolean mask array where True indicates dithering area
    """
    # Convert fractions to pixel coordinates
    cx = center_x * width
    cy = center_y * height

    # Use average of width and height for radius calculation
    r_pixels = radius * (width + height) / 2.0

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate distances from center
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Create mask
    mask = distances <= r_pixels

    return mask


def create_gradient_mask(
    width: int,
    height: int,
    split_ratio: float,
    cut_direction: Literal['vertical', 'horizontal'],
    fade_min: float
) -> npt.NDArray[np.float64]:
    """
    Create a gradient mask for fade-out effect in cut modes.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        split_ratio: Position of the cut (0.0 to 1.0)
        cut_direction: 'vertical' or 'horizontal'
        fade_min: Minimum density at the far edge (0.0 to 1.0)

    Returns:
        Float mask array with values from fade_min to 1.0
    """
    mask = np.zeros((height, width), dtype=np.float64)

    if cut_direction == 'vertical':
        split_pos = int(width * split_ratio)
        dither_width = width - split_pos

        if dither_width > 0:
            # Create gradient from 1.0 at cut line to fade_min at right edge
            gradient = np.linspace(1.0, fade_min, dither_width)
            mask[:, split_pos:] = gradient[np.newaxis, :]
    else:  # horizontal
        split_pos = int(height * split_ratio)
        dither_height = height - split_pos

        if dither_height > 0:
            # Create gradient from 1.0 at cut line to fade_min at bottom edge
            gradient = np.linspace(1.0, fade_min, dither_height)
            mask[split_pos:, :] = gradient[:, np.newaxis]

    return mask


def create_gradient_density_mask(
    width: int,
    height: int,
    angle: float,
    density_start: float,
    density_end: float
) -> npt.NDArray[np.float64]:
    """
    Create a gradient density mask that transitions from one density to another.

    Uses angle-based gradients where the gradient transitions across the image
    based on the specified angle.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        angle: Gradient angle in degrees (0-360)
            - 0°: left to right
            - 90°: top to bottom
            - 180°: right to left
            - 270°: bottom to top
        density_start: Density at start (0.0 to 1.0)
        density_end: Density at end (0.0 to 1.0)

    Returns:
        Float mask array with values transitioning from density_start to density_end
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Create directional vector (cos, sin)
    # Note: In image coordinates, y increases downward
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Create coordinate grids (normalized to [0, 1])
    y_coords, x_coords = np.ogrid[:height, :width]
    x_norm = x_coords.astype(float) / max(width - 1, 1)
    y_norm = y_coords.astype(float) / max(height - 1, 1)

    # Project each pixel position onto the gradient direction
    # This gives us a value that increases along the gradient direction
    projection = x_norm * dx + y_norm * dy

    # Normalize projection to [0, 1] range
    # The projection range depends on the angle
    proj_min = projection.min()
    proj_max = projection.max()

    if proj_max > proj_min:
        projection_normalized = (projection - proj_min) / (proj_max - proj_min)
    else:
        projection_normalized = np.zeros_like(projection)

    # Map to [density_start, density_end]
    mask = density_start + (density_end - density_start) * projection_normalized

    return mask


def glitch_swap_rows(
    image_array: npt.NDArray[np.integer],
    intensity: float,
    seed: Optional[int] = None
) -> npt.NDArray[np.integer]:
    """
    Randomly swap rows in the image array based on intensity.

    Args:
        image_array: Numpy array (2D or 3D)
        intensity: Glitch intensity (0.0 to 1.0)
        seed: Random seed

    Returns:
        Modified array
    """
    rng = np.random.default_rng(seed)
    height = image_array.shape[0]

    # Determine number of swaps based on intensity (max 50% of rows for 1.0)
    num_swaps = int(height * intensity * 0.5)

    # Optimization: Use indices array to perform swaps instead of moving large image rows
    indices = np.arange(height)

    if num_swaps > 0:
        # Generate all random indices at once
        # We need 2 * num_swaps integers
        # Interleave to match the RNG consumption order of the original loop
        # Loop was: y1 = rng(), y2 = rng()
        random_indices = rng.integers(0, height, size=2 * num_swaps)
        y1s = random_indices[0::2]
        y2s = random_indices[1::2]

        # Perform swaps on indices array
        for i in range(num_swaps):
            y1, y2 = y1s[i], y2s[i]
            indices[y1], indices[y2] = indices[y2], indices[y1]

    # Apply the shuffled indices to the image using advanced indexing
    return image_array[indices]


def ordered_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int,
    matrix: npt.NDArray[np.float64],
    threshold_offset: float = 0.0,
    density_mask: Optional[npt.NDArray[np.float64]] = None,
    seed: Optional[int] = None  # Unused but kept for interface consistency
) -> npt.NDArray[np.uint8]:
    """
    Apply ordered dithering using a threshold matrix.
    """
    height, width = image_array.shape
    mh, mw = matrix.shape

    # Tile the matrix to cover the image
    tiled_matrix = np.tile(matrix, (height // mh + 1, width // mw + 1))
    tiled_matrix = tiled_matrix[:height, :width]

    # Calculate effective threshold for each pixel
    # Map matrix (0-1) to (0-255) and shift by user threshold
    # Base threshold 128 -> center.
    # We want: pixel > matrix_val * 255 + (threshold - 128)

    threshold_shift = threshold - 128.0 + threshold_offset
    effective_thresholds = (tiled_matrix * 255.0) + threshold_shift

    # Start with original image values
    result = image_array.astype(np.uint8).copy()

    # Create dithering mask (where to apply dithering)
    dither_mask_pixels = np.ones((height, width), dtype=bool)

    if density_mask is not None:
        # Mark pixels outside dithered region (density_mask == 0) to preserve
        outside_region = density_mask == 0.0
        dither_mask_pixels[outside_region] = False

        # For pixels inside the region, apply probabilistic fade
        rng = np.random.default_rng(seed=seed)
        density_random = rng.uniform(0.0, 1.0, size=(height, width))
        # Skip pixels where random > density (fade effect)
        skip_mask = density_random > density_mask
        # But only within the dithered region
        skip_mask = skip_mask & ~outside_region
    else:
        skip_mask = np.zeros((height, width), dtype=bool)

    # Determine which pixels should be white (255) vs black (0)
    white_pixels = (image_array > effective_thresholds) & dither_mask_pixels
    black_pixels = (image_array <= effective_thresholds) & dither_mask_pixels

    # Apply dithering only where dither_mask_pixels is True
    result[white_pixels | skip_mask] = 255
    result[black_pixels & ~skip_mask] = 0

    return result


def atkinson_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int = 128,
    randomize: bool = True,
    jitter: float = 15.0,
    threshold_offset: float = 0.0,
    seed: Optional[int] = None,
    density_mask: Optional[npt.NDArray[np.float64]] = None,
    satoshi_mode: bool = False
) -> npt.NDArray[np.uint8]:
    """
    Apply Atkinson dithering algorithm.
    Propagates error to 6 neighbors with 1/8 weight.
    """
    img = image_array.astype(float)
    height, width = img.shape
    rng = np.random.default_rng(seed=seed)

    random_noise: npt.NDArray[np.float64]
    if randomize:
        random_noise = rng.uniform(-jitter, jitter, size=(height, width))
    else:
        random_noise = np.zeros((height, width))

    density_random: Optional[npt.NDArray[np.float64]] = None
    if density_mask is not None:
        density_random = rng.uniform(0.0, 1.0, size=(height, width))

    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            adjusted_threshold = threshold + random_noise[y, x] + threshold_offset

            if density_mask is not None and density_random is not None:
                if density_mask[y, x] == 0.0:
                    # Outside dithered region - preserve original pixel
                    continue
                elif density_random[y, x] > density_mask[y, x]:
                    # Within dithered region but skipped for fade
                    img[y, x] = 255.0
                    continue
                else:
                    new_pixel = 255 if old_pixel > adjusted_threshold else 0
            else:
                new_pixel = 255 if old_pixel > adjusted_threshold else 0

            img[y, x] = new_pixel
            error = old_pixel - new_pixel

            # Atkinson distribution (1/8 to neighbors)
            #       X   1   1
            #   1   1   1
            #       1
            fraction = 1.0 / 8.0

            # (x+1, y)
            if x + 1 < width:
                img[y, x + 1] += error * fraction
            # (x+2, y)
            if x + 2 < width:
                img[y, x + 2] += error * fraction
            # (x-1, y+1)
            if y + 1 < height and x > 0:
                img[y + 1, x - 1] += error * fraction
            # (x, y+1)
            if y + 1 < height:
                img[y + 1, x] += error * fraction
            # (x+1, y+1)
            if y + 1 < height and x + 1 < width:
                img[y + 1, x + 1] += error * fraction
            # (x, y+2)
            if y + 2 < height:
                img[y + 2, x] += error * fraction

    return (img > 128).astype(np.uint8) * 255


def hal_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int = 128,
    threshold_offset: float = 0.0,
    seed: Optional[int] = None,
    density_mask: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray[np.uint8]:
    """
    Apply 'Hal' dithering (tribute to Hal Finney).
    Simulates PGP-era terminal with scanline effects and subtle noise.
    """
    img = image_array.astype(float)
    height, width = img.shape
    rng = np.random.default_rng(seed=seed)

    # Create scanline effect: modify threshold based on Y coordinate
    # Alternating lines or every 4th line darker/lighter
    y_coords, x_coords = np.indices((height, width))

    # Scanline pattern: sine wave
    scanline_pattern = np.sin(y_coords * 0.8) * 40.0

    # Digital noise (subtle)
    noise = rng.normal(0, 20.0, size=(height, width))

    # Combine
    adjusted_thresholds = threshold + scanline_pattern + noise + threshold_offset

    # Start with original image values
    result = image_array.astype(np.uint8).copy()

    # Density mask check
    should_dither = np.ones((height, width), dtype=bool)
    if density_mask is not None:
        # Pixels outside dithered region (density_mask == 0) should not be modified
        outside_region = density_mask == 0.0
        # For pixels inside region, apply probabilistic fade
        density_random = rng.uniform(0.0, 1.0, size=(height, width))
        skip_fade = (density_random > density_mask) & ~outside_region
        should_dither = ~outside_region & ~skip_fade

        # Skipped pixels (fade effect) within region become white
        result[skip_fade] = 255

    # Apply dithering where should_dither is True
    white_pixels = (img > adjusted_thresholds) & should_dither
    black_pixels = (img <= adjusted_thresholds) & should_dither

    result[white_pixels] = 255
    result[black_pixels] = 0

    return result


@jit(nopython=True)
def _floyd_steinberg_jit(
    img: npt.NDArray[np.float64],
    original_img: npt.NDArray[np.integer],
    random_noise: npt.NDArray[np.float64],
    threshold: float,
    threshold_offset: float,
    density_mask: npt.NDArray[np.float64],
    density_random: npt.NDArray[np.float64],
    use_mask: bool,
    satoshi_mode: bool
) -> None:
    """Core Floyd-Steinberg loop optimized with Numba."""
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]

            # Calculate base threshold
            base_threshold = threshold
            if satoshi_mode:
                # Dynamic threshold based on image brightness (Satoshi Mode)
                # Brighter areas get higher threshold -> fewer red pixels
                original_pixel = float(original_img[y, x])
                base_threshold += (original_pixel - 128.0) * 0.5

            # Apply randomized threshold
            adjusted_threshold = base_threshold + random_noise[y, x] + threshold_offset

            # Check density mask - probabilistically skip pixels for fade effect
            if use_mask:
                if density_mask[y, x] == 0.0:
                    # Outside dithered region - preserve original pixel
                    continue
                elif density_random[y, x] > density_mask[y, x]:
                    # Within dithered region but skipped for fade - force to white
                    img[y, x] = 255.0
                    continue  # Skip error diffusion for this pixel
                else:
                    new_pixel = 255.0 if old_pixel > adjusted_threshold else 0.0
            else:
                new_pixel = 255.0 if old_pixel > adjusted_threshold else 0.0

            img[y, x] = new_pixel

            error = old_pixel - new_pixel

            # Distribute error to neighboring pixels (Floyd-Steinberg pattern)
            if x + 1 < width:
                img[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                if x > 0:
                    img[y + 1, x - 1] += error * 3 / 16
                img[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    img[y + 1, x + 1] += error * 1 / 16


def floyd_steinberg_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int = 128,
    randomize: bool = True,
    jitter: float = 15.0,
    threshold_offset: float = 0.0,
    seed: Optional[int] = None,
    density_mask: Optional[npt.NDArray[np.float64]] = None,
    satoshi_mode: bool = False
) -> npt.NDArray[np.uint8]:
    """
    Apply Floyd-Steinberg dithering algorithm with optional randomization.

    Randomization adds small noise to the threshold to break up regular patterns
    and create more organic-looking dithered results, preventing visual artifacts.

    Args:
        image_array: Grayscale numpy array
        threshold: Base threshold for binary conversion (0-255)
        randomize: Add random noise to threshold to reduce artifacts (default: True)
        jitter: Amount of random noise to add (±jitter). Default: 15.0
        threshold_offset: Bias added to threshold. Positive = darker (more red). Default: 0.0
        seed: Random seed for reproducible results.
        density_mask: Optional mask controlling dithering density (0.0 to 1.0).
                     Values < 1.0 probabilistically skip pixels for fade effects.
        satoshi_mode: Enable dynamic threshold based on local brightness.

    Returns:
        Binary dithered array
    """
    # Make a copy to avoid modifying original
    img = image_array.astype(float)
    height, width = img.shape

    # Initialize random number generator for reproducible randomness
    rng = np.random.default_rng(seed=seed)

    # Generate random threshold adjustments if randomization is enabled
    # Small random values (±15) are added to threshold to break up patterns
    random_noise: npt.NDArray[np.float64]
    if randomize:
        random_noise = rng.uniform(-jitter, jitter, size=(height, width))
    else:
        random_noise = np.zeros((height, width))

    # Generate random values for density mask if provided
    use_mask = False
    # Use 2D dummy arrays to match type expected by JIT function
    density_mask_arr = np.zeros((1, 1), dtype=float)
    density_random_arr = np.zeros((1, 1), dtype=float)

    if density_mask is not None:
        use_mask = True
        density_mask_arr = density_mask
        density_random_arr = rng.uniform(0.0, 1.0, size=(height, width))

    # Call JIT-optimized core function
    _floyd_steinberg_jit(
        img,
        image_array,
        random_noise,
        float(threshold),
        threshold_offset,
        density_mask_arr,
        density_random_arr,
        use_mask,
        satoshi_mode
    )

    return (img > 128).astype(np.uint8) * 255


def get_output_filename(input_path: Union[str, Path]) -> Path:
    """
    Generate output filename with -dither suffix, avoiding overwrites.

    Args:
        input_path: Path to input image

    Returns:
        Path object for output file
    """
    path = Path(input_path)
    stem = path.stem
    suffix = path.suffix
    directory = path.parent

    # Start with base name
    output_path = directory / f"{stem}-dither{suffix}"

    # If file exists, append number
    counter = 1
    while output_path.exists():
        output_path = directory / f"{stem}-dither-{counter}{suffix}"
        counter += 1

    return output_path


def apply_dithering_algorithm(
    pattern: DitherPattern,
    image_array: npt.NDArray[np.integer],
    threshold: int,
    randomize: bool,
    jitter: float,
    threshold_offset: float,
    seed: Optional[int],
    density_mask: Optional[npt.NDArray[np.float64]],
    satoshi_mode: bool = False
) -> npt.NDArray[np.uint8]:
    """
    Dispatch to appropriate dithering function.
    """
    if pattern == 'floyd-steinberg':
        return floyd_steinberg_dither(
            image_array, threshold, randomize, jitter, threshold_offset, seed, density_mask, satoshi_mode
        )
    elif pattern == 'atkinson':
        return atkinson_dither(
            image_array, threshold, randomize, jitter, threshold_offset, seed, density_mask
        )
    elif pattern == 'ordered':
        return ordered_dither(
            image_array, threshold, BAYER_8x8, threshold_offset, density_mask, seed
        )
    elif pattern == 'clustered-dot':
        return ordered_dither(
            image_array, threshold, CLUSTERED_8x8, threshold_offset, density_mask, seed
        )
    elif pattern == 'bitcoin':
        return ordered_dither(
            image_array, threshold, BITCOIN_8x8, threshold_offset, density_mask, seed
        )
    elif pattern == 'hal':
        return hal_dither(
            image_array, threshold, threshold_offset, seed, density_mask
        )
    else:
        # Fallback to FS
        return floyd_steinberg_dither(
            image_array, threshold, randomize, jitter, threshold_offset, seed, density_mask, satoshi_mode
        )


def apply_dither(
    img: Image.Image,
    split_ratio: Optional[float] = None,
    cut_direction: Literal['vertical', 'horizontal'] = 'vertical',
    threshold: int = 128,
    grayscale_original: bool = False,
    randomize: bool = True,
    jitter: float = 15.0,
    reference_width: int = 1024,
    darkness_offset: float = 0.0,
    seed: Optional[int] = None,
    rectangles: Optional[list[tuple[float, float, float, float]]] = None,
    circles: Optional[list[tuple[float, float, float]]] = None,
    fade: Optional[float] = None,
    gradient: Optional[tuple[float, float, float]] = None,
    background: Literal['white', 'dark'] = 'white',
    pattern: DitherPattern = 'floyd-steinberg',
    satoshi_mode: bool = False,
    brand: str = 'btcat',
    glitch: float = 0.0,
    shade: str = "1"
) -> Image.Image:
    """
    Apply dithering to a PIL Image using brand colors and selected pattern.
    """
    if split_ratio is None:
        # Golden ratio split: original side is ~38%, dithered side is ~62%
        split_ratio = 1 / GOLDEN_RATIO

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Apply grayscale to entire image first if requested
    if grayscale_original:
        img = img.convert('L').convert('RGB')

    width, height = img.size

    # Keep a copy of the base image for non-dithered areas
    base_img = img.copy()

    # Handle RGB brand vs Monochrome brands
    brand_config = BRANDS.get(brand, BRANDS['btcat'])
    is_rgb_mode = brand_config['type'] == 'rgb'

    # Calculate scale factor based on total image width
    scale_factor = 1.0
    if reference_width > 0 and width > reference_width:
        scale_factor = width / reference_width

    original_size = (width, height)

    # Prepare image for dithering (Grayscale or separate RGB channels)
    if is_rgb_mode:
        # Split into channels
        channels = img.split() # R, G, B
        channel_arrays = []
        for channel in channels:
            if scale_factor > 1.0:
                new_size = (int(width / scale_factor), int(height / scale_factor))
                new_size = (max(1, new_size[0]), max(1, new_size[1]))
                channel = channel.resize(new_size, Image.Resampling.LANCZOS)
            channel_arrays.append(np.array(channel))

        # Determine dimensions from first channel (all should be same)
        h, w = channel_arrays[0].shape
        # Initialize img_array to avoid unbound error (though unused in rgb mode)
        img_array = np.zeros((h, w), dtype=np.uint8)
    else:
        # Convert to grayscale for dithering
        img_gray = img.convert('L')
        if scale_factor > 1.0:
            new_size = (int(width / scale_factor), int(height / scale_factor))
            new_size = (max(1, new_size[0]), max(1, new_size[1]))
            img_gray = img_gray.resize(new_size, Image.Resampling.LANCZOS)

        img_array = np.array(img_gray)
        h, w = img_array.shape

    # If no shapes specified, create default rectangle for cut mode (backward compatibility)
    if not rectangles and not circles:
        rectangles = []
        if cut_direction == 'vertical':
            # Vertical cut: from split position to right edge
            rectangles.append((split_ratio, 0.0, 1.0, 1.0))
        else:  # horizontal
            # Horizontal cut: from split position to bottom edge
            rectangles.append((0.0, split_ratio, 1.0, 1.0))

    # Create combined mask from all shapes
    dither_mask = np.zeros((h, w), dtype=bool)

    # Add all rectangles
    if rectangles:
        for x1, y1, x2, y2 in rectangles:
            rect_mask = create_rectangle_mask(w, h, x1, y1, x2, y2)
            dither_mask = np.logical_or(dither_mask, rect_mask)

    # Add all circles
    if circles:
        for cx, cy, r in circles:
            circle_mask = create_circle_mask(w, h, cx, cy, r)
            dither_mask = np.logical_or(dither_mask, circle_mask)

    # Apply gradient or uniform fade if specified (applies to all dithered areas)
    density_mask: Optional[npt.NDArray[np.float64]] = None

    # Gradient takes precedence over uniform fade
    if gradient is not None:
        # Unpack gradient tuple (angle, start, end)
        angle, gradient_start, gradient_end = gradient
        # Create gradient density mask across entire image
        density_mask = create_gradient_density_mask(w, h, angle, gradient_start, gradient_end)
        # Only apply where dither_mask is True
        density_mask = np.where(dither_mask, density_mask, 0.0)
    elif fade is not None:
        # Create a uniform density mask for fade effect
        density_mask = np.full((h, w), fade, dtype=np.float64)
        # Only apply where dither_mask is True
        density_mask = np.where(dither_mask, density_mask, 0.0)

    # Apply dithering using selected pattern
    # Glitch Mode: Repeated error diffusion passes with noise feedback
    if is_rgb_mode:
        # RGB mode: dither each channel separately
        dithered_channels = []
        for i, channel_array in enumerate(channel_arrays): # type: ignore
            # Determine number of passes for glitch mode
            passes = 1 + int(glitch * 3) if glitch > 0.0 else 1

            if passes > 1:
                # Multi-pass glitch mode
                rng = np.random.default_rng(seed)
                current_array = channel_array.copy().astype(np.int16)
                d_array = np.zeros_like(channel_array, dtype=np.uint8)

                for p in range(passes):
                    input_for_pass = current_array.astype(int) if p == 0 else d_array.astype(int)

                    if p > 0:
                        # Add noise to corrupt the feedback loop
                        noise_amount = int(50 * glitch)
                        if noise_amount > 0:
                            noise = rng.integers(-noise_amount, noise_amount + 1, size=input_for_pass.shape)
                            input_for_pass = np.clip(input_for_pass + noise, 0, 255)

                    pass_seed = seed + p if seed is not None else None
                    d_array = apply_dithering_algorithm(
                        pattern, input_for_pass, threshold, randomize, jitter, darkness_offset, pass_seed, density_mask, satoshi_mode
                    )
            else:
                # Normal single-pass dithering
                d_array = apply_dithering_algorithm(
                    pattern, channel_array, threshold, randomize, jitter, darkness_offset, seed, density_mask, satoshi_mode
                )

            dithered_channels.append(d_array)

        # Combine channels
        dithered_array = np.dstack(dithered_channels) # (h, w, 3)
    else:
        # Monochrome mode: single channel dithering
        passes = 1 + int(glitch * 3) if glitch > 0.0 else 1

        if passes > 1:
            # Multi-pass glitch mode
            rng = np.random.default_rng(seed)
            current_img_array = img_array.copy().astype(np.int16)
            dithered_array = np.zeros((h, w), dtype=np.uint8)

            for p in range(passes):
                input_for_pass = current_img_array.astype(int) if p == 0 else dithered_array.astype(int)

                if p > 0:
                    # Add noise to corrupt the feedback loop
                    noise_amount = int(50 * glitch)
                    if noise_amount > 0:
                        noise = rng.integers(-noise_amount, noise_amount + 1, size=(h, w))
                        input_for_pass = np.clip(input_for_pass + noise, 0, 255)

                pass_seed = seed + p if seed is not None else None
                dithered_array = apply_dithering_algorithm(
                    pattern, input_for_pass, threshold, randomize, jitter, darkness_offset, pass_seed, density_mask, satoshi_mode
                )
        else:
            # Normal single-pass dithering
            dithered_array = apply_dithering_algorithm(
                pattern, img_array, threshold, randomize, jitter, darkness_offset, seed, density_mask, satoshi_mode
            )

    # Upscale if needed
    if scale_factor > 1.0:
        if is_rgb_mode:
             dithered_img_temp = Image.fromarray(dithered_array.astype(np.uint8))
             dithered_img_temp = dithered_img_temp.resize(original_size, Image.Resampling.NEAREST)
             dithered_array = np.array(dithered_img_temp)
        else:
            dithered_img_temp = Image.fromarray(dithered_array.astype(np.uint8))
            dithered_img_temp = dithered_img_temp.resize(original_size, Image.Resampling.NEAREST)
            dithered_array = np.array(dithered_img_temp)

        # Also upscale the mask
        if dither_mask is not None:
            mask_img = Image.fromarray((dither_mask * 255).astype(np.uint8))
            mask_img = mask_img.resize(original_size, Image.Resampling.NEAREST)
            dither_mask = np.array(mask_img) > 128

    # Glitch Mode: Row Swapping
    if glitch > 0.0:
        # Swap rows on the final dithered bitmap
        glitch_seed = seed if seed is not None else np.random.randint(0, 10000)
        dithered_array = glitch_swap_rows(dithered_array, glitch, seed=glitch_seed)

    # Create RGB image with brand colors
    # Background color depends on mode: white or dark
    bg_color = DARK_BACKGROUND if background == 'dark' else (255, 255, 255)
    dithered_rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Re-initialize RNG for channel shift (glitch mode)
    rng = np.random.default_rng(seed)

    if is_rgb_mode:
        # RGB mode: map 255 (background) to bg_color, and 0 (ink) to 0
        for i in range(3):
            channel_data = np.where(dithered_array[:, :, i] == 255, bg_color[i], 0)

            # Glitch Mode: Channel Shifting (RGB mode)
            if glitch > 0.0:
                max_shift = int(width * glitch * 0.05)
                if max_shift > 0:
                    shift = rng.integers(-max_shift, max_shift + 1)
                    channel_data = np.roll(channel_data, shift, axis=1)

            dithered_rgb[:, :, i] = channel_data
    else:
        # Monochrome mode: use brand color
        brand_color = brand_config.get('color', (0,0,0))

        # Parse shade parameter
        shade_factor = 0.0
        shade_quant: Optional[int] = None
        if shade:
            try:
                parts = str(shade).split(',')
                shade_factor = float(parts[0])
                for part in parts[1:]:
                    part = part.strip()
                    if part.startswith('q='):
                        shade_quant = int(part.split('=')[1])
            except ValueError:
                pass

        # Calculate shading map if needed
        t_map: Optional[npt.NDArray[np.float64]] = None
        if shade_factor > 0.0:
            # Normalize grayscale image to 0.0-1.0
            t_map = img_array.astype(float) / 255.0

            # Apply quantization if requested
            if shade_quant is not None and shade_quant > 1:
                t_map = np.round(t_map * (shade_quant - 1)) / (shade_quant - 1)

            # Apply shade factor
            t_map = t_map * shade_factor

            # Upscale t_map to match original size if needed
            if scale_factor > 1.0:
                # Use PIL 'F' mode for floating point image resizing
                t_map_img = Image.fromarray(t_map.astype(np.float32))
                t_map_img = t_map_img.resize(original_size, Image.Resampling.NEAREST)
                t_map = np.array(t_map_img)

            # Apply glitch row swapping to t_map if enabled
            if glitch > 0.0:
                glitch_seed = seed if seed is not None else np.random.randint(0, 10000)
                # t_map is float, glitch_swap_rows expects int but works with copy
                # Cast to mimic glitch behavior or adapt function?
                # glitch_swap_rows handles copy. Let's cast to object or specific type if needed
                # Actually glitch_swap_rows uses .copy() and assignment. It should work for float arrays too
                # provided the type hint doesn't crash runtime (it won't).
                t_map = glitch_swap_rows(t_map, glitch, seed=glitch_seed) # type: ignore

        for i in range(3):
            if t_map is not None:
                # Interpolate between brand color and background color based on t_map
                # t_map = 0 -> Brand Color
                # t_map = 1 -> Background Color
                brand_c = float(brand_color[i])
                bg_c = float(bg_color[i])

                dot_colors = brand_c * (1.0 - t_map) + bg_c * t_map
                dot_colors = np.clip(dot_colors, 0, 255).astype(np.uint8)

                channel_data = np.where(dithered_array == 0, dot_colors, bg_color[i])
            else:
                channel_data = np.where(dithered_array == 0, brand_color[i], bg_color[i])

            # Glitch Mode: Channel Shifting (monochrome mode)
            if glitch > 0.0:
                max_shift = int(width * glitch * 0.05)
                if max_shift > 0:
                    shift = rng.integers(-max_shift, max_shift + 1)
                    channel_data = np.roll(channel_data, shift, axis=1)

            dithered_rgb[:, :, i] = channel_data

    # Create result image by compositing
    result_array = np.array(base_img)

    # Apply dithering only where mask is True
    if dither_mask is not None:
        for i in range(3):
            result_array[:, :, i] = np.where(dither_mask, dithered_rgb[:, :, i], result_array[:, :, i])

    result = Image.fromarray(result_array)
    return result


def dither_image(
    input_path: Union[str, Path],
    split_ratio: Optional[float] = None,
    cut_direction: Literal['vertical', 'horizontal'] = 'vertical',
    threshold: int = 128,
    grayscale_original: bool = False,
    randomize: bool = True,
    jitter: float = 15.0,
    reference_width: int = 1024,
    darkness_offset: float = 0.0,
    seed: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    rectangles: Optional[list[tuple[float, float, float, float]]] = None,
    circles: Optional[list[tuple[float, float, float]]] = None,
    fade: Optional[float] = None,
    gradient: Optional[tuple[float, float, float]] = None,
    background: Literal['white', 'dark'] = 'white',
    pattern: DitherPattern = 'floyd-steinberg',
    satoshi_mode: bool = False,
    brand: str = 'btcat',
    glitch: float = 0.0,
    shade: str = "1"
) -> Path:
    """
    Apply dithering to a portion of an image using brand colors.

    Args:
        input_path: Path to input image file
        split_ratio: Position for the cut (0.0 to 1.0, default: golden ratio ~0.382)
        cut_direction: 'vertical' or 'horizontal' (default: 'vertical')
        threshold: Threshold for dithering (0-255, default: 128)
        grayscale_original: Convert the original (non-dithered) part to grayscale (default: False)
        randomize: Add random noise to threshold to reduce regular patterns (default: True)
        jitter: Amount of random noise to add.
        reference_width: Target width for scaling point size.
        darkness_offset: Bias for darkness.
        seed: Random seed for reproducible results.
        output_path: Optional path for output file. If None, generated from input filename.
        rectangles: List of rectangles [(x1, y1, x2, y2), ...]. Coordinates can be any float value.
        circles: List of circles [(cx, cy, r), ...]. Coordinates can be any float value.
        fade: Dithering density (0.0 to 1.0). Controls sparsity of dithering across all areas.
              1.0 = full density, 0.1 = only 10% of pixels dithered.
        background: Background color for dithered areas. 'white' (default) or 'dark' (#222222).
        pattern: Dithering pattern to use.
        satoshi_mode: Enable dynamic threshold based on local brightness.
        brand: Brand palette to use ('btcat', 'lightning', 'cypherpunk', 'rgb').
        glitch: Glitch intensity (0.0 to 1.0). Enables row swapping, channel shifting, and repeated passes.
        shade: Shade factor and quantization (e.g., "1", "0.5", "0.5,q=3"). Default "1".
               Controls how the brand dots are shaded based on original grayscale value.

    Returns:
        Path to output file
    """
    # Load image
    try:
        img = Image.open(input_path)
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    result = apply_dither(
        img,
        split_ratio=split_ratio,
        cut_direction=cut_direction,
        threshold=threshold,
        grayscale_original=grayscale_original,
        randomize=randomize,
        jitter=jitter,
        reference_width=reference_width,
        darkness_offset=darkness_offset,
        seed=seed,
        rectangles=rectangles,
        circles=circles,
        fade=fade,
        gradient=gradient,
        background=background,
        pattern=pattern,
        satoshi_mode=satoshi_mode,
        brand=brand,
        glitch=glitch,
        shade=shade
    )

    # Determine final output path
    final_output_path: Path
    if output_path is None:
        final_output_path = get_output_filename(input_path)
    else:
        final_output_path = Path(output_path)

    # Save with appropriate format
    if final_output_path.suffix.lower() in ['.jpg', '.jpeg']:
        result.save(final_output_path, 'JPEG', quality=95)
    elif final_output_path.suffix.lower() == '.png':
        result.save(final_output_path, 'PNG')
    else:
        result.save(final_output_path)

    return final_output_path


@click.command()
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--cut',
    type=click.Choice(['vertical', 'horizontal'], case_sensitive=False),
    default='vertical',
    show_default=True,
    help='Cut direction: vertical (left/right) or horizontal (top/bottom)'
)
@click.option(
    '--pos',
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help='Position of the cut (0.0 to 1.0). Default: golden ratio ~0.382'
)
@click.option(
    '--threshold',
    type=click.IntRange(0, 255),
    default=128,
    show_default=True,
    help='Threshold for dithering (0-255)'
)
@click.option(
    '--pattern',
    type=click.Choice(['floyd-steinberg', 'ordered', 'atkinson', 'clustered-dot', 'bitcoin', 'hal'], case_sensitive=False),
    default='floyd-steinberg',
    show_default=True,
    help='Dithering pattern to use.'
)
@click.option(
    '--grayscale',
    is_flag=True,
    help='Convert entire image to grayscale before applying dithering'
)
@click.option(
    '--no-randomize',
    is_flag=True,
    help='Disable threshold randomization (produces more regular patterns)'
)
@click.option(
    '--jitter',
    type=click.FloatRange(0.0, 255.0),
    default=30.0,
    show_default=True,
    help='Amount of random noise to add to threshold.'
)
@click.option(
    '--reference-width',
    type=int,
    default=1024,
    show_default=True,
    help='Reference width for point size scaling. Dithering point size will be proportional to image width relative to this.'
)
@click.option(
    '--darkness',
    type=float,
    default=0.0,
    show_default=True,
    help='Adjust darkness (draws less background pixels). Positive values make it darker.'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for reproducible results.',
    hidden=False
)
@click.option(
    '--output', '-o',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='Output file path. Defaults to automatic naming.'
)
@click.option(
    '--rect',
    multiple=True,
    help='Rectangle region: x1,y1,x2,y2 (fractions, can be outside 0-1). Can be specified multiple times.'
)
@click.option(
    '--circle',
    multiple=True,
    help='Circle region: x,y,radius (fractions, can be outside 0-1). Can be specified multiple times.'
)
@click.option(
    '--fade',
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help='Dithering density (0.0 to 1.0). E.g., 0.1 = only 10%% of pixels dithered (sparse effect). Applies to all dithered areas.'
)
@click.option(
    '--gradient',
    default=None,
    help='Gradient density: angle,start,end (e.g., "0,0.1,1.0"). Angle: 0-360° (0=left→right, 90=top→bottom, 180=right→left, 270=bottom→top). Start/End: density 0.0-1.0. Overrides --fade.'
)
@click.option(
    '--background',
    type=click.Choice(['white', 'dark'], case_sensitive=False),
    default='white',
    show_default=True,
    help='Background color for dithered areas. "white" for white background, "dark" for dark gray (#222222).'
)
@click.option(
    '--satoshi-mode',
    is_flag=True,
    help='Enable Satoshi Mode: Dynamic threshold based on local brightness.'
)
@click.option(
    '--brand',
    type=click.Choice(list(BRANDS.keys()), case_sensitive=False),
    default='btcat',
    show_default=True,
    help='Brand palette to use.'
)
@click.option(
    '--glitch',
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    help='Glitch Mode intensity (0.0 to 1.0). Adds row swapping, channel shifting, and repeated passes.'
)
@click.option(
    '--shade',
    type=str,
    default="1",
    show_default=True,
    help='Shade factor and quantization (e.g. "1", "0.5", "0.5,q=3"). Controls intensity of dot shading based on grayscale value.'
)
def main(
    image: str,
    cut: Literal['vertical', 'horizontal'],
    pos: Optional[float],
    threshold: int,
    pattern: DitherPattern,
    grayscale: bool,
    no_randomize: bool,
    jitter: float,
    reference_width: int,
    darkness: float,
    seed: Optional[int],
    output: Optional[str],
    rect: tuple[str, ...],
    circle: tuple[str, ...],
    fade: Optional[float],
    gradient: Optional[str],
    background: Literal['white', 'dark'],
    satoshi_mode: bool,
    brand: str,
    glitch: float,
    shade: str
) -> None:
    """Apply monochrome dithering to a portion of an image using brand colors.

    IMAGE is the path to the input image file (PNG or JPG).

    By default, randomization is applied to the dithering threshold to create
    more organic, less regular patterns. Use --no-randomize for classic
    Floyd-Steinberg without randomization.

    Dithering modes:

    - Default: Vertical cut at golden ratio (can be changed with --cut and --pos)

    - Rectangles: Specify one or more with --rect=x1,y1,x2,y2
      Example: --rect=0,0,0.1,1 --rect=0.9,0,1,1 (left and right strips)

    - Circles: Specify one or more with --circle=x,y,radius
      Example: --circle=0.5,0.5,0.2 (centered circle)

    - Mix: Combine rectangles and circles
      Example: --rect=0,0,1,0.1 --circle=0.5,0.5,0.2

    Options apply globally:
    - --pattern: Choose algorithm (floyd-steinberg, ordered, atkinson, clustered-dot, bitcoin, hal)
    - --grayscale: Converts entire image to grayscale before applying dithering
    - --fade: Controls dithering density (0.1 = sparse/10%, 1.0 = full density)
    - --satoshi-mode: Adapts threshold per pixel based on local brightness
    - --glitch: Adds glitch effects (row swapping, channel shift, repeated passes)
    """
    try:
        # Parse rectangle specifications
        rectangles: Optional[list[tuple[float, float, float, float]]] = None
        if rect:
            rectangles = []
            for r in rect:
                try:
                    parts = [float(x.strip()) for x in r.split(',')]
                    if len(parts) != 4:
                        raise ValueError(f"Rectangle must have 4 values (x1,y1,x2,y2), got {len(parts)}")
                    rectangles.append((parts[0], parts[1], parts[2], parts[3]))
                except ValueError as e:
                    click.secho(f"Error parsing rectangle '{r}': {e}", fg='red', err=True)
                    sys.exit(1)

        # Parse circle specifications
        circles: Optional[list[tuple[float, float, float]]] = None
        if circle:
            circles = []
            for c in circle:
                try:
                    parts = [float(x.strip()) for x in c.split(',')]
                    if len(parts) != 3:
                        raise ValueError(f"Circle must have 3 values (x,y,radius), got {len(parts)}")
                    circles.append((parts[0], parts[1], parts[2]))
                except ValueError as e:
                    click.secho(f"Error parsing circle '{c}': {e}", fg='red', err=True)
                    sys.exit(1)

        # Parse gradient specification
        gradient_tuple: Optional[tuple[float, float, float]] = None
        if gradient:
            try:
                parts = [float(x.strip()) for x in gradient.split(',')]
                if len(parts) != 3:
                    raise ValueError(f"Gradient must have 3 values (angle,start,end), got {len(parts)}")
                angle, grad_start, grad_end = parts[0], parts[1], parts[2]

                # Validate ranges
                if not (0.0 <= grad_start <= 1.0):
                    raise ValueError(f"Gradient start must be between 0.0 and 1.0, got {grad_start}")
                if not (0.0 <= grad_end <= 1.0):
                    raise ValueError(f"Gradient end must be between 0.0 and 1.0, got {grad_end}")

                gradient_tuple = (angle, grad_start, grad_end)
            except ValueError as e:
                click.secho(f"Error parsing gradient '{gradient}': {e}", fg='red', err=True)
                sys.exit(1)

        output_path = dither_image(
            image,
            split_ratio=pos,
            cut_direction=cut,
            threshold=threshold,
            grayscale_original=grayscale,
            randomize=not no_randomize,
            jitter=jitter,
            reference_width=reference_width,
            darkness_offset=darkness,
            seed=seed,
            output_path=output,
            rectangles=rectangles,
            circles=circles,
            fade=fade,
            gradient=gradient_tuple,
            background=background,
            pattern=pattern,
            satoshi_mode=satoshi_mode,
            brand=brand,
            glitch=glitch,
            shade=shade
        )
        click.secho(f"✓ Dithered image saved to: {output_path}", fg='green')
    except Exception as e:
        click.secho(f"Error: {e}", fg='red', err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
