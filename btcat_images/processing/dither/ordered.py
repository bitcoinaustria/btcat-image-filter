import numpy as np
import numpy.typing as npt
from typing import Optional

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
