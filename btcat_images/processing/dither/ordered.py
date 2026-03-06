import numpy as np
import numpy.typing as npt
from typing import Optional
from numba import njit, prange

@njit(parallel=True, cache=True)
def _ordered_jit(
    image_array: npt.NDArray[np.integer],
    matrix: npt.NDArray[np.float64],
    threshold_shift: float,
    noise: npt.NDArray[np.float64],
    density_mask: npt.NDArray[np.float64],
    density_random: npt.NDArray[np.float64],
    use_mask: bool
) -> npt.NDArray[np.uint8]:
    height, width = image_array.shape
    mh, mw = matrix.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for y in prange(height):
        my = y % mh
        for x in range(width):
            mx = x % mw
            effective_threshold = (matrix[my, mx] * 255.0) + threshold_shift + noise[y, x]

            should_dither = True
            is_skipped = False

            if use_mask:
                if density_mask[y, x] == 0.0:
                    should_dither = False
                elif density_random[y, x] > density_mask[y, x]:
                    is_skipped = True

            if should_dither:
                if is_skipped:
                    result[y, x] = 255
                else:
                    if image_array[y, x] > effective_threshold:
                        result[y, x] = 255
                    else:
                        result[y, x] = 0
            else:
                result[y, x] = np.uint8(image_array[y, x])

    return result

def ordered_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int,
    matrix: npt.NDArray[np.float64],
    threshold_offset: float = 0.0,
    density_mask: Optional[npt.NDArray[np.float64]] = None,
    seed: Optional[int] = None,
    jitter: float = 0.0
) -> npt.NDArray[np.uint8]:
    """
    Apply ordered dithering using a threshold matrix (Bayer, Clustered Dot, etc.).

    This algorithm compares each pixel against a value in a tiled threshold matrix.
    It produces very structured, grid-like patterns.

    Args:
        image_array: Grayscale numpy array (2D).
        threshold: Base threshold level (0-255). 128 is neutral.
        matrix: 2D float array (values 0.0-1.0) representing the dither pattern.
        threshold_offset: Bias added to threshold. Positive = darker output. Default: 0.0.
        density_mask: Optional mask (0.0-1.0) for fade effects.
        seed: Random seed for reproducible jitter and density mask fade.
        jitter: Amount of random noise (±jitter) to add to thresholds. Default: 0.0.

    Returns:
        Binary dithered array (uint8).
    """
    height, width = image_array.shape
    rng = np.random.default_rng(seed=seed)

    threshold_shift = float(threshold) - 128.0 + threshold_offset

    noise = np.zeros((height, width), dtype=np.float64)
    if jitter > 0.0:
        noise = rng.uniform(-jitter, jitter, size=(height, width))

    use_mask = False
    density_mask_arr = np.zeros((1, 1), dtype=np.float64)
    density_random = np.zeros((1, 1), dtype=np.float64)

    if density_mask is not None:
        use_mask = True
        density_mask_arr = density_mask
        density_random = rng.uniform(0.0, 1.0, size=(height, width))

    return _ordered_jit(
        image_array,
        matrix,
        threshold_shift,
        noise,
        density_mask_arr,
        density_random,
        use_mask
    )