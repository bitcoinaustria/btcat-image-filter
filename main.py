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


# Austrian flag red color
AUSTRIAN_RED: Tuple[int, int, int] = (237, 41, 57)  # #ED2939

# Golden ratio (phi)
GOLDEN_RATIO: float = 1.618033988749895


def create_rectangle_mask(
    width: int,
    height: int,
    bl_x: float,
    bl_y: float,
    tr_x: float,
    tr_y: float
) -> npt.NDArray[np.bool_]:
    """
    Create a rectangular mask for dithering.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        bl_x: Bottom-left X coordinate (0.0 to 1.0)
        bl_y: Bottom-left Y coordinate (0.0 to 1.0)
        tr_x: Top-right X coordinate (0.0 to 1.0)
        tr_y: Top-right Y coordinate (0.0 to 1.0)

    Returns:
        Boolean mask array where True indicates dithering area
    """
    mask = np.zeros((height, width), dtype=bool)

    # Convert fractions to pixel coordinates
    # Y coordinates: 0.0 = top, 1.0 = bottom (inverted for image coordinates)
    x1 = int(bl_x * width)
    y1 = int((1.0 - tr_y) * height)  # tr_y is top, invert
    x2 = int(tr_x * width)
    y2 = int((1.0 - bl_y) * height)  # bl_y is bottom, invert

    # Ensure coordinates are within bounds
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    # Fill the rectangle
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True

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


def floyd_steinberg_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int = 128,
    randomize: bool = True,
    jitter: float = 15.0,
    threshold_offset: float = 0.0,
    seed: Optional[int] = None,
    density_mask: Optional[npt.NDArray[np.float64]] = None
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
    density_random: Optional[npt.NDArray[np.float64]] = None
    if density_mask is not None:
        density_random = rng.uniform(0.0, 1.0, size=(height, width))

    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            # Apply randomized threshold to reduce regular patterns
            adjusted_threshold = threshold + random_noise[y, x] + threshold_offset

            # Check density mask - probabilistically skip pixels for fade effect
            should_dither = True
            if density_mask is not None and density_random is not None:
                if density_random[y, x] > density_mask[y, x]:
                    # Skip this pixel, force to white
                    should_dither = False
                    new_pixel = 255.0
                else:
                    new_pixel = 255 if old_pixel > adjusted_threshold else 0
            else:
                new_pixel = 255 if old_pixel > adjusted_threshold else 0

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

    return (img > threshold).astype(np.uint8) * 255


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
    mode: Literal['cut', 'rect', 'circle'] = 'cut',
    rect_bl_x: float = 0.2,
    rect_bl_y: float = 0.2,
    rect_tr_x: float = 0.8,
    rect_tr_y: float = 0.8,
    circle_x: float = 0.5,
    circle_y: float = 0.5,
    circle_r: float = 0.3,
    fade: Optional[float] = None
) -> Path:
    """
    Apply dithering to a portion of an image using Austrian flag red.

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
        mode: Dithering mode: 'cut' (default), 'rect' (rectangle), or 'circle'
        rect_bl_x: Rectangle bottom-left X (0.0 to 1.0, for 'rect' mode)
        rect_bl_y: Rectangle bottom-left Y (0.0 to 1.0, for 'rect' mode)
        rect_tr_x: Rectangle top-right X (0.0 to 1.0, for 'rect' mode)
        rect_tr_y: Rectangle top-right Y (0.0 to 1.0, for 'rect' mode)
        circle_x: Circle center X (0.0 to 1.0, for 'circle' mode)
        circle_y: Circle center Y (0.0 to 1.0, for 'circle' mode)
        circle_r: Circle radius (0.0 to 1.0, for 'circle' mode)
        fade: Fade-out intensity for 'cut' mode (0.0 to 1.0). If set, creates gradient.

    Returns:
        Path to output file
    """
    if split_ratio is None:
        # Golden ratio split: original side is ~38%, dithered side is ~62%
        split_ratio = 1 / GOLDEN_RATIO

    # Load image
    try:
        img = Image.open(input_path)
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size

    # Create the base for grayscale conversion if needed
    if grayscale_original:
        grayscale_img = img.convert('L').convert('RGB')
    else:
        grayscale_img = img.copy()

    # Convert entire image to grayscale for dithering
    img_gray = img.convert('L')

    # Calculate scale factor based on total image width
    scale_factor = 1.0
    if reference_width > 0 and width > reference_width:
        scale_factor = width / reference_width

    # Scale down if needed
    original_size = (width, height)
    if scale_factor > 1.0:
        new_size = (int(width / scale_factor), int(height / scale_factor))
        new_size = (max(1, new_size[0]), max(1, new_size[1]))
        img_gray = img_gray.resize(new_size, Image.Resampling.LANCZOS)

    img_array = np.array(img_gray)

    # Create mask and density mask based on mode
    dither_mask: Optional[npt.NDArray[np.bool_]] = None
    density_mask: Optional[npt.NDArray[np.float64]] = None

    if mode == 'cut':
        # Create mask for cut mode
        h, w = img_array.shape
        dither_mask = np.zeros((h, w), dtype=bool)

        if cut_direction == 'vertical':
            split_pos = int(w * split_ratio)
            dither_mask[:, split_pos:] = True
        else:  # horizontal
            split_pos = int(h * split_ratio)
            dither_mask[split_pos:, :] = True

        # Add gradient fade if specified
        if fade is not None:
            density_mask = create_gradient_mask(w, h, split_ratio, cut_direction, fade)

    elif mode == 'rect':
        # Create rectangular mask
        h, w = img_array.shape
        dither_mask = create_rectangle_mask(w, h, rect_bl_x, rect_bl_y, rect_tr_x, rect_tr_y)

    elif mode == 'circle':
        # Create circular mask
        h, w = img_array.shape
        dither_mask = create_circle_mask(w, h, circle_x, circle_y, circle_r)

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'cut', 'rect', or 'circle'.")

    # Apply Floyd-Steinberg dithering
    dithered_array = floyd_steinberg_dither(
        img_array, threshold, randomize, jitter, darkness_offset, seed, density_mask
    )

    # Upscale if needed
    if scale_factor > 1.0:
        dithered_img_temp = Image.fromarray(dithered_array.astype(np.uint8))
        dithered_img_temp = dithered_img_temp.resize(original_size, Image.Resampling.NEAREST)
        dithered_array = np.array(dithered_img_temp)

        # Also upscale the mask
        if dither_mask is not None:
            mask_img = Image.fromarray((dither_mask * 255).astype(np.uint8))
            mask_img = mask_img.resize(original_size, Image.Resampling.NEAREST)
            dither_mask = np.array(mask_img) > 128

    # Create RGB image with Austrian red for dithered pixels
    dithered_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(3):
        dithered_rgb[:, :, i] = np.where(dithered_array == 0, AUSTRIAN_RED[i], 255)

    # Create result image by compositing
    result_array = np.array(grayscale_img)

    # Apply dithering only where mask is True
    if dither_mask is not None:
        for i in range(3):
            result_array[:, :, i] = np.where(dither_mask, dithered_rgb[:, :, i], result_array[:, :, i])

    result = Image.fromarray(result_array)

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
    '--grayscale',
    is_flag=True,
    help='Convert the original (non-dithered) part to grayscale'
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
    '--mode',
    type=click.Choice(['cut', 'rect', 'circle'], case_sensitive=False),
    default='cut',
    show_default=True,
    help='Dithering mode: cut (default), rect (rectangle), or circle'
)
@click.option(
    '--rect-bl-x',
    type=click.FloatRange(0.0, 1.0),
    default=0.2,
    show_default=True,
    help='Rectangle mode: bottom-left X coordinate (0.0 to 1.0)'
)
@click.option(
    '--rect-bl-y',
    type=click.FloatRange(0.0, 1.0),
    default=0.2,
    show_default=True,
    help='Rectangle mode: bottom-left Y coordinate (0.0 to 1.0)'
)
@click.option(
    '--rect-tr-x',
    type=click.FloatRange(0.0, 1.0),
    default=0.8,
    show_default=True,
    help='Rectangle mode: top-right X coordinate (0.0 to 1.0)'
)
@click.option(
    '--rect-tr-y',
    type=click.FloatRange(0.0, 1.0),
    default=0.8,
    show_default=True,
    help='Rectangle mode: top-right Y coordinate (0.0 to 1.0)'
)
@click.option(
    '--circle-x',
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help='Circle mode: center X coordinate (0.0 to 1.0)'
)
@click.option(
    '--circle-y',
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help='Circle mode: center Y coordinate (0.0 to 1.0)'
)
@click.option(
    '--circle-r',
    type=click.FloatRange(0.0, 1.0),
    default=0.3,
    show_default=True,
    help='Circle mode: radius (0.0 to 1.0, fraction of image dimensions)'
)
@click.option(
    '--fade',
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help='Cut mode: fade-out intensity (0.0 to 1.0). E.g., 0.1 = fade to 10%% density at edge'
)
def main(
    image: str,
    cut: Literal['vertical', 'horizontal'],
    pos: Optional[float],
    threshold: int,
    grayscale: bool,
    no_randomize: bool,
    jitter: float,
    reference_width: int,
    darkness: float,
    seed: Optional[int],
    output: Optional[str],
    mode: Literal['cut', 'rect', 'circle'],
    rect_bl_x: float,
    rect_bl_y: float,
    rect_tr_x: float,
    rect_tr_y: float,
    circle_x: float,
    circle_y: float,
    circle_r: float,
    fade: Optional[float]
) -> None:
    """Apply monochrome dithering to a portion of an image using Austrian flag red.

    IMAGE is the path to the input image file (PNG or JPG).

    By default, randomization is applied to the dithering threshold to create
    more organic, less regular patterns. Use --no-randomize for classic
    Floyd-Steinberg without randomization.

    Supports three dithering modes:
    - cut: Traditional cut mode (vertical or horizontal)
    - rect: Rectangle mode (dither within a rectangular region)
    - circle: Circle mode (dither within a circular region)
    """
    try:
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
            mode=mode,
            rect_bl_x=rect_bl_x,
            rect_bl_y=rect_bl_y,
            rect_tr_x=rect_tr_x,
            rect_tr_y=rect_tr_y,
            circle_x=circle_x,
            circle_y=circle_y,
            circle_r=circle_r,
            fade=fade
        )
        click.secho(f"✓ Dithered image saved to: {output_path}", fg='green')
    except Exception as e:
        click.secho(f"Error: {e}", fg='red', err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
