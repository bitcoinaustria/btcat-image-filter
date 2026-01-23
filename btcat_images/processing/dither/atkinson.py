import numpy as np
import numpy.typing as npt
from numba import jit
from typing import Optional

@jit(nopython=True)
def _atkinson_jit(
    img: npt.NDArray[np.float64],
    random_noise: npt.NDArray[np.float64],
    threshold: float,
    threshold_offset: float,
    density_mask: npt.NDArray[np.float64],
    density_random: npt.NDArray[np.float64],
    use_mask: bool
) -> None:
    """Core Atkinson loop optimized with Numba."""
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            adjusted_threshold = threshold + random_noise[y, x] + threshold_offset

            if use_mask:
                if density_mask[y, x] == 0.0:
                    # Outside dithered region - preserve original pixel
                    continue
                elif density_random[y, x] > density_mask[y, x]:
                    # Within dithered region but skipped for fade
                    img[y, x] = 255.0
                    continue
                else:
                    new_pixel = 255.0 if old_pixel > adjusted_threshold else 0.0
            else:
                new_pixel = 255.0 if old_pixel > adjusted_threshold else 0.0

            img[y, x] = new_pixel
            error = old_pixel - new_pixel

            # Atkinson distribution (1/8 to neighbors)
            #       X   1   1
            #   1   1   1
            #       1
            fraction = 1.0 / 8.0

            # (x+1, y)
            if x + 1 < width:
                img[y, x + 1] += error * fraction
            # (x+2, y)
            if x + 2 < width:
                img[y, x + 2] += error * fraction
            # (x-1, y+1)
            if y + 1 < height and x > 0:
                img[y + 1, x - 1] += error * fraction
            # (x, y+1)
            if y + 1 < height:
                img[y + 1, x] += error * fraction
            # (x+1, y+1)
            if y + 1 < height and x + 1 < width:
                img[y + 1, x + 1] += error * fraction
            # (x, y+2)
            if y + 2 < height:
                img[y + 2, x] += error * fraction

def atkinson_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int = 128,
    randomize: bool = True,
    jitter: float = 15.0,
    threshold_offset: float = 0.0,
    seed: Optional[int] = None,
    density_mask: Optional[npt.NDArray[np.float64]] = None,
    satoshi_mode: bool = False # Unused in Atkinson but kept for signature consistency
) -> npt.NDArray[np.uint8]:
    """
    Apply Atkinson dithering algorithm.
    Propagates error to 6 neighbors with 1/8 weight.
    """
    img = image_array.astype(float)
    height, width = img.shape
    rng = np.random.default_rng(seed=seed)

    random_noise: npt.NDArray[np.float64]
    if randomize:
        random_noise = rng.uniform(-jitter, jitter, size=(height, width))
    else:
        random_noise = np.zeros((height, width))

    use_mask = False
    density_mask_arr = np.zeros((1, 1), dtype=float)
    density_random_arr = np.zeros((1, 1), dtype=float)

    if density_mask is not None:
        use_mask = True
        density_mask_arr = density_mask
        density_random_arr = rng.uniform(0.0, 1.0, size=(height, width))

    _atkinson_jit(
        img,
        random_noise,
        float(threshold),
        threshold_offset,
        density_mask_arr,
        density_random_arr,
        use_mask
    )

    return (img > 128).astype(np.uint8) * 255
