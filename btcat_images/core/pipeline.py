from pathlib import Path
from typing import Optional, Union, Literal
from PIL import Image
import numpy as np
import numpy.typing as npt

from ..constants import BRANDS, DARK_BACKGROUND, GOLDEN_RATIO, DitherPattern
from ..processing.masks import create_rectangle_mask, create_circle_mask, create_gradient_density_mask
from ..processing.filters.glitch import glitch_swap_rows
from ..processing.dither import apply_dithering_algorithm
from .utils import get_output_filename

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
            dither_mask |= rect_mask

    # Add all circles
    if circles:
        for cx, cy, r in circles:
            circle_mask = create_circle_mask(w, h, cx, cy, r)
            dither_mask |= circle_mask

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
