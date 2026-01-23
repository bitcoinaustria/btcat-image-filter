import numpy as np
import numpy.typing as npt
from numba import jit
from typing import Optional

@jit(nopython=True)
def _floyd_steinberg_jit(
    img: npt.NDArray[np.float64],
    original_img: npt.NDArray[np.integer],
    random_noise: npt.NDArray[np.float64],
    threshold: float,
    threshold_offset: float,
    density_mask: npt.NDArray[np.float64],
    density_random: npt.NDArray[np.float64],
    use_mask: bool,
    satoshi_mode: bool
) -> None:
    """
    Core Floyd-Steinberg error diffusion loop optimized with Numba.

    This function performs the pixel-by-pixel dithering and error propagation.
    It is compiled with Numba for high performance.

    Args:
        img: Float array of the image to modify in-place.
        original_img: Original integer image array (used for Satoshi mode).
        random_noise: Pre-computed noise array for threshold randomization.
        threshold: Base threshold value.
        threshold_offset: Global offset to apply to the threshold.
        density_mask: Array controlling pixel density (0.0 to 1.0).
        density_random: Pre-computed random values for density checks.
        use_mask: Boolean flag indicating if density_mask should be used.
        satoshi_mode: Boolean flag to enable brightness-adaptive thresholding.
    """
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]

            # Calculate base threshold
            base_threshold = threshold
            if satoshi_mode:
                # Dynamic threshold based on image brightness (Satoshi Mode)
                # Brighter areas get higher threshold -> fewer red pixels
                original_pixel = float(original_img[y, x])
                base_threshold += (original_pixel - 128.0) * 0.5

            # Apply randomized threshold
            adjusted_threshold = base_threshold + random_noise[y, x] + threshold_offset

            # Check density mask - probabilistically skip pixels for fade effect
            if use_mask:
                if density_mask[y, x] == 0.0:
                    # Outside dithered region - preserve original pixel
                    continue
                elif density_random[y, x] > density_mask[y, x]:
                    # Within dithered region but skipped for fade - force to white
                    img[y, x] = 255.0
                    continue  # Skip error diffusion for this pixel
                else:
                    new_pixel = 255.0 if old_pixel > adjusted_threshold else 0.0
            else:
                new_pixel = 255.0 if old_pixel > adjusted_threshold else 0.0

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


def floyd_steinberg_dither(
    image_array: npt.NDArray[np.integer],
    threshold: int = 128,
    randomize: bool = True,
    jitter: float = 15.0,
    threshold_offset: float = 0.0,
    seed: Optional[int] = None,
    density_mask: Optional[npt.NDArray[np.float64]] = None,
    satoshi_mode: bool = False
) -> npt.NDArray[np.uint8]:
    """
    Apply Floyd-Steinberg dithering algorithm with optional randomization.

    Randomization adds small noise to the threshold to break up regular patterns
    and create more organic-looking dithered results, preventing visual artifacts.

    Args:
        image_array: Grayscale numpy array (2D).
        threshold: Base threshold for binary conversion (0-255).
        randomize: If True, adds random noise to threshold to reduce artifacts.
        jitter: Amount of random noise to add (±jitter). Default: 15.0.
        threshold_offset: Bias added to threshold. Positive = darker output. Default: 0.0.
        seed: Random seed for reproducible results.
        density_mask: Optional mask (0.0-1.0) where values < 1.0 probabilistically
                      skip pixels to create fade effects.
        satoshi_mode: If True, enables dynamic thresholding based on local brightness.

    Returns:
        Binary dithered array (uint8) where 0 is black and 255 is white.
    """
    # Make a copy to avoid modifying original
    img = image_array.astype(float)
    height, width = img.shape

    # Initialize random number generator for reproducible randomness
    rng = np.random.default_rng(seed=seed)

    # Generate random threshold adjustments if randomization is enabled
    # Small random values (±15) are added to threshold to break up patterns
    random_noise: npt.NDArray[np.float64]
    if randomize:
        random_noise = rng.uniform(-jitter, jitter, size=(height, width))
    else:
        random_noise = np.zeros((height, width))

    # Generate random values for density mask if provided
    use_mask = False
    # Use 2D dummy arrays to match type expected by JIT function
    density_mask_arr = np.zeros((1, 1), dtype=float)
    density_random_arr = np.zeros((1, 1), dtype=float)

    if density_mask is not None:
        use_mask = True
        density_mask_arr = density_mask
        density_random_arr = rng.uniform(0.0, 1.0, size=(height, width))

    # Call JIT-optimized core function
    _floyd_steinberg_jit(
        img,
        image_array,
        random_noise,
        float(threshold),
        threshold_offset,
        density_mask_arr,
        density_random_arr,
        use_mask,
        satoshi_mode
    )

    return (img > 128).astype(np.uint8) * 255