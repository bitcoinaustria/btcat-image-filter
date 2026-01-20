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


# Austrian flag red color
AUSTRIAN_RED = (237, 41, 57)  # #ED2939

# Golden ratio (phi)
GOLDEN_RATIO = 1.618033988749895


def floyd_steinberg_dither(image_array, threshold=128, randomize=True):
    """
    Apply Floyd-Steinberg dithering algorithm with optional randomization.

    Randomization adds small noise to the threshold to break up regular patterns
    and create more organic-looking dithered results, preventing visual artifacts.

    Args:
        image_array: Grayscale numpy array
        threshold: Base threshold for binary conversion (0-255)
        randomize: Add random noise to threshold to reduce artifacts (default: True)

    Returns:
        Binary dithered array
    """
    # Make a copy to avoid modifying original
    img = image_array.astype(float)
    height, width = img.shape

    # Initialize random number generator for reproducible randomness
    rng = np.random.default_rng(seed=None)

    # Generate random threshold adjustments if randomization is enabled
    # Small random values (±15) are added to threshold to break up patterns
    if randomize:
        random_noise = rng.uniform(-15, 15, size=(height, width))
    else:
        random_noise = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            # Apply randomized threshold to reduce regular patterns
            adjusted_threshold = threshold + random_noise[y, x]
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


def get_output_filename(input_path):
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


def dither_image(input_path, split_ratio=None, cut_direction='vertical', threshold=128, grayscale_original=False, randomize=True):
    """
    Apply dithering to a portion of an image using Austrian flag red.

    Args:
        input_path: Path to input image file
        split_ratio: Position for the cut (0.0 to 1.0, default: golden ratio ~0.382)
        cut_direction: 'vertical' or 'horizontal' (default: 'vertical')
        threshold: Threshold for dithering (0-255, default: 128)
        grayscale_original: Convert the original (non-dithered) part to grayscale (default: False)
        randomize: Add random noise to threshold to reduce regular patterns (default: True)

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

    if cut_direction == 'vertical':
        # Vertical cut: left part is original, right part is dithered
        split_pos = int(width * split_ratio)

        # Split the image
        original_part = img.crop((0, 0, split_pos, height))
        dither_part = img.crop((split_pos, 0, width, height))

        # Convert original part to grayscale if requested
        if grayscale_original:
            original_part = original_part.convert('L').convert('RGB')

        # Convert dither part to grayscale
        dither_gray = dither_part.convert('L')
        dither_array = np.array(dither_gray)

        # Apply Floyd-Steinberg dithering
        dithered_array = floyd_steinberg_dither(dither_array, threshold, randomize)

        # Create RGB image with Austrian red using broadcasting
        dithered_rgb = np.full((height, width - split_pos, 3), 255, dtype=np.uint8)
        mask = dithered_array == 0
        dithered_rgb[mask] = AUSTRIAN_RED

        dithered_part_img = Image.fromarray(dithered_rgb)

        # Combine parts
        result = Image.new('RGB', (width, height))
        result.paste(original_part, (0, 0))
        result.paste(dithered_part_img, (split_pos, 0))

    elif cut_direction == 'horizontal':
        # Horizontal cut: top part is original, bottom part is dithered
        split_pos = int(height * split_ratio)

        # Split the image
        original_part = img.crop((0, 0, width, split_pos))
        dither_part = img.crop((0, split_pos, width, height))

        # Convert original part to grayscale if requested
        if grayscale_original:
            original_part = original_part.convert('L').convert('RGB')

        # Convert dither part to grayscale
        dither_gray = dither_part.convert('L')
        dither_array = np.array(dither_gray)

        # Apply Floyd-Steinberg dithering
        dithered_array = floyd_steinberg_dither(dither_array, threshold, randomize)

        # Create RGB image with Austrian red using broadcasting
        dithered_rgb = np.full((height - split_pos, width, 3), 255, dtype=np.uint8)
        mask = dithered_array == 0
        dithered_rgb[mask] = AUSTRIAN_RED

        dithered_part_img = Image.fromarray(dithered_rgb)

        # Combine parts
        result = Image.new('RGB', (width, height))
        result.paste(original_part, (0, 0))
        result.paste(dithered_part_img, (0, split_pos))
    else:
        raise ValueError(f"Invalid cut_direction: {cut_direction}. Must be 'vertical' or 'horizontal'.")

    # Generate output filename
    output_path = get_output_filename(input_path)

    # Save with appropriate format
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        result.save(output_path, 'JPEG', quality=95)
    elif output_path.suffix.lower() == '.png':
        result.save(output_path, 'PNG')
    else:
        result.save(output_path)

    return output_path


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
def main(image, cut, pos, threshold, grayscale, no_randomize):
    """Apply monochrome dithering to a portion of an image using Austrian flag red.

    IMAGE is the path to the input image file (PNG or JPG).

    By default, randomization is applied to the dithering threshold to create
    more organic, less regular patterns. Use --no-randomize for classic
    Floyd-Steinberg without randomization.
    """
    try:
        output_path = dither_image(
            image,
            split_ratio=pos,
            cut_direction=cut,
            threshold=threshold,
            grayscale_original=grayscale,
            randomize=not no_randomize
        )
        click.secho(f"✓ Dithered image saved to: {output_path}", fg='green')
    except Exception as e:
        click.secho(f"Error: {e}", fg='red', err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
