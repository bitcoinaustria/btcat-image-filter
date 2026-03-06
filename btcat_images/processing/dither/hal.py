import numpy as np
import numpy.typing as npt
from typing import Optional
from numba import njit, prange

@njit(parallel=True, cache=True)
def _hal_jit(
    img: npt.NDArray[np.float64],
    threshold: float,
    threshold_offset: float,
    noise: npt.NDArray[np.float64],
    density_mask: npt.NDArray[np.float64],
    density_random: npt.NDArray[np.float64],
    use_mask: bool
) -> npt.NDArray[np.uint8]:
    height, width = img.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for y in prange(height):
        # Calculate scanline pattern for this row
        scanline = np.sin(y * 0.8) * 40.0
        for x in range(width):
            adjusted_threshold = threshold + scanline + noise[y, x] + threshold_offset

            should_dither = True
            if use_mask:
                if density_mask[y, x] == 0.0:
                    should_dither = False
                    result[y, x] = np.uint8(img[y, x])
                elif density_random[y, x] > density_mask[y, x]:
                    should_dither = False
                    result[y, x] = 255

            if should_dither:
                if img[y, x] > adjusted_threshold:
                    result[y, x] = 255
                else:
                    result[y, x] = 0

    return result

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
    img = image_array.astype(np.float64)
    height, width = img.shape
    rng = np.random.default_rng(seed=seed)

    # Digital noise (subtle)
    noise = rng.normal(0, 20.0, size=(height, width))

    use_mask = False
    density_mask_arr = np.zeros((1, 1), dtype=np.float64)
    density_random = np.zeros((1, 1), dtype=np.float64)

    if density_mask is not None:
        use_mask = True
        density_mask_arr = density_mask
        density_random = rng.uniform(0.0, 1.0, size=(height, width))

    return _hal_jit(
        img,
        float(threshold),
        threshold_offset,
        noise,
        density_mask_arr,
        density_random,
        use_mask
    )