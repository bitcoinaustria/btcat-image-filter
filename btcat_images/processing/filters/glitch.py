import numpy as np
import numpy.typing as npt
from typing import Optional
from numba import njit

@njit(cache=True)
def _glitch_indices_jit(height: int, y1s: npt.NDArray[np.int64], y2s: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    indices = np.arange(height)
    num_swaps = len(y1s)
    for i in range(num_swaps):
        y1, y2 = y1s[i], y2s[i]
        tmp = indices[y1]
        indices[y1] = indices[y2]
        indices[y2] = tmp
    return indices

def glitch_swap_rows(
    image_array: npt.NDArray[np.integer],
    intensity: float,
    seed: Optional[int] = None
) -> npt.NDArray[np.integer]:
    """
    Randomly swap rows in the image array to create a glitch effect.

    This function simulates data corruption by vertically swapping lines of pixels.
    The number of swaps is proportional to the image height and the intensity.

    Args:
        image_array: Numpy array (2D grayscale or 3D RGB).
        intensity: Glitch intensity (0.0 to 1.0). 1.0 swaps up to 50% of rows.
        seed: Random seed for reproducible glitches.

    Returns:
        Modified image array with swapped rows.
    """
    rng = np.random.default_rng(seed)
    height = image_array.shape[0]

    # Determine number of swaps based on intensity (max 50% of rows for 1.0)
    num_swaps = int(height * intensity * 0.5)

    if num_swaps > 0:
        # Generate random indices
        random_indices = rng.integers(0, height, size=2 * num_swaps)
        y1s = random_indices[0::2]
        y2s = random_indices[1::2]

        # Perform swaps on indices array using JIT
        indices = _glitch_indices_jit(height, y1s, y2s)

        # Apply the shuffled indices to the image using advanced indexing
        return image_array[indices]

    return image_array