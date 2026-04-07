import numpy as np
from numpy.typing import NDArray
from PIL import Image
from numba import njit, prange

from ...constants import ORIGINAL_PALETTE


@njit(cache=True)
def _find_closest_color(r: float, g: float, b: float, palette: NDArray) -> int:
    """Find index of closest color in palette using Euclidean distance.

    Optimized: Unrolled loop for exactly 3 colors (as per original mode design).
    """
    dr0 = r - palette[0, 0]
    dg0 = g - palette[0, 1]
    db0 = b - palette[0, 2]
    dist0 = dr0 * dr0 + dg0 * dg0 + db0 * db0

    dr1 = r - palette[1, 0]
    dg1 = g - palette[1, 1]
    db1 = b - palette[1, 2]
    dist1 = dr1 * dr1 + dg1 * dg1 + db1 * db1

    dr2 = r - palette[2, 0]
    dg2 = g - palette[2, 1]
    db2 = b - palette[2, 2]
    dist2 = dr2 * dr2 + dg2 * dg2 + db2 * db2

    if dist0 <= dist1 and dist0 <= dist2:
        return 0
    elif dist1 <= dist2:
        return 1
    else:
        return 2


@njit(cache=True)
def _floyd_steinberg_3color(
    pixels: NDArray, 
    width: int, 
    height: int, 
    palette: NDArray
) -> NDArray:
    """3-color Floyd-Steinberg error diffusion on RGB data.

    Operates on a float32 (h*w, 4) array in-place, distributing quantization
    error to neighboring pixels using the classic 7/16, 3/16, 5/16, 1/16 weights.
    """
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            old_r = pixels[idx, 0]
            old_g = pixels[idx, 1]
            old_b = pixels[idx, 2]

            ci = _find_closest_color(old_r, old_g, old_b, palette)
            new_r = palette[ci, 0]
            new_g = palette[ci, 1]
            new_b = palette[ci, 2]

            pixels[idx, 0] = new_r
            pixels[idx, 1] = new_g
            pixels[idx, 2] = new_b

            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b

            # Distribute error to neighbors
            if x + 1 < width:
                ni = y * width + (x + 1)
                pixels[ni, 0] += err_r * 7.0 / 16.0
                pixels[ni, 1] += err_g * 7.0 / 16.0
                pixels[ni, 2] += err_b * 7.0 / 16.0

            if y + 1 < height:
                if x - 1 >= 0:
                    ni = (y + 1) * width + (x - 1)
                    pixels[ni, 0] += err_r * 3.0 / 16.0
                    pixels[ni, 1] += err_g * 3.0 / 16.0
                    pixels[ni, 2] += err_b * 3.0 / 16.0

                ni = (y + 1) * width + x
                pixels[ni, 0] += err_r * 5.0 / 16.0
                pixels[ni, 1] += err_g * 5.0 / 16.0
                pixels[ni, 2] += err_b * 5.0 / 16.0

                if x + 1 < width:
                    ni = (y + 1) * width + (x + 1)
                    pixels[ni, 0] += err_r * 1.0 / 16.0
                    pixels[ni, 1] += err_g * 1.0 / 16.0
                    pixels[ni, 2] += err_b * 1.0 / 16.0

    return pixels


def original_dither(
    image_rgb: NDArray,
    palette: list[tuple[int, int, int]] | None = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    detail: float = 1.0,
    point_size: int = 4,
) -> NDArray:
    """Apply 3-color Floyd-Steinberg dithering (nbb original mode).

    Pipeline:
    1. Detail reduction (blur via downscale/upscale if detail < 1.0)
    2. Downscale by point_size factor
    3. Brightness/contrast adjustments
    4. Floyd-Steinberg error diffusion with 3-color palette
    5. Upscale back to original size with nearest-neighbor

    Args:
        image_rgb: Input RGB image as (h, w, 3) uint8 array
        palette: List of RGB tuples for dithering palette. Defaults to ORIGINAL_PALETTE.
        brightness: Brightness multiplier (0.0-2.0, default 1.0)
        contrast: Contrast multiplier (0.0-2.0, default 1.0)
        detail: Detail level (0.6-1.0, default 1.0). Lower = more blur.
        point_size: Pixel block size (1-8, default 4). Higher = chunkier pixels.

    Returns:
        Dithered RGB image as (h, w, 3) uint8 array
    """
    if palette is None:
        palette = ORIGINAL_PALETTE

    h, w = image_rgb.shape[:2]
    img = Image.fromarray(image_rgb)

    # Step 1: Detail reduction (blur effect via downscale-upscale)
    work_w = max(1, w // point_size)
    work_h = max(1, h // point_size)

    if detail < 1.0:
        detail_w = max(1, round(work_w * detail))
        detail_h = max(1, round(work_h * detail))
        img_detail = img.resize((detail_w, detail_h), Image.Resampling.LANCZOS)
        img_work = img_detail.resize((work_w, work_h), Image.Resampling.LANCZOS)
    else:
        img_work = img.resize((work_w, work_h), Image.Resampling.LANCZOS)

    # Step 2: Get pixel data as float32 for error diffusion
    work_array = np.array(img_work, dtype=np.float32)
    wh, ww = work_array.shape[:2]

    # Step 3: Brightness/contrast adjustments (matching JS implementation)
    if brightness != 1.0:
        work_array[:, :, :3] *= brightness

    if contrast != 1.0:
        contrast_val = contrast / 5.0
        factor = (259.0 * (contrast_val * 255.0 + 255.0)) / (255.0 * (259.0 - contrast_val * 255.0))
        work_array[:, :, :3] = factor * (work_array[:, :, :3] - 128.0) + 128.0

    work_array = np.clip(work_array, 0.0, 255.0)

    # Step 4: Floyd-Steinberg 3-color dithering
    palette_array = np.array(palette, dtype=np.float32)

    # Reshape to (h*w, 3) for the numba function — add alpha channel placeholder
    pixels_flat = np.zeros((wh * ww, 4), dtype=np.float32)
    pixels_flat[:, :3] = work_array[:, :, :3].reshape(-1, 3)
    pixels_flat[:, 3] = 255.0

    pixels_flat = _floyd_steinberg_3color(pixels_flat, ww, wh, palette_array)

    # Clamp and reshape back
    result = np.clip(pixels_flat[:, :3], 0.0, 255.0).reshape(wh, ww, 3).astype(np.uint8)

    # Step 5: Upscale with nearest-neighbor for pixel look
    out_w = ww * point_size
    out_h = wh * point_size
    result_img = Image.fromarray(result)
    result_img = result_img.resize((out_w, out_h), Image.Resampling.NEAREST)

    return np.array(result_img)
