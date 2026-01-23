import numpy as np
import numpy.typing as npt
from typing import Optional

def hal_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int = 128,
    threshold_offset: float = 0.0,
    seed: Optional[int] = None,
    density_mask: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray[np.uint8]:
    """
    Apply 'Hal' dithering (tribute to Hal Finney).

    This custom algorithm simulates a PGP-era terminal aesthetic with
    scanline effects (sine wave based) and subtle digital noise.

    Args:
        image_array: Grayscale numpy array (2D).
        threshold: Base threshold level (0-255).
        threshold_offset: Bias added to threshold. Positive = darker output.
        seed: Random seed for noise generation.
        density_mask: Optional mask (0.0-1.0) for fade effects.

    Returns:
        Binary dithered array (uint8).
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