import numpy as np
import numpy.typing as npt
from typing import Optional

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
