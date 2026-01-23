from typing import Optional
import numpy as np
import numpy.typing as npt
from ...constants import DitherPattern, BAYER_8x8, CLUSTERED_8x8, BITCOIN_8x8

from .floyd_steinberg import floyd_steinberg_dither
from .ordered import ordered_dither
from .atkinson import atkinson_dither
from .hal import hal_dither

def apply_dithering_algorithm(
    pattern: DitherPattern,
    image_array: npt.NDArray[np.integer],
    threshold: int,
    randomize: bool,
    jitter: float,
    threshold_offset: float,
    seed: Optional[int],
    density_mask: Optional[npt.NDArray[np.float64]],
    satoshi_mode: bool = False
) -> npt.NDArray[np.uint8]:
    """
    Dispatch to appropriate dithering function.
    """
    match pattern:
        case 'floyd-steinberg':
            return floyd_steinberg_dither(
                image_array, threshold, randomize, jitter, threshold_offset, seed, density_mask, satoshi_mode
            )
        case 'atkinson':
            return atkinson_dither(
                image_array, threshold, randomize, jitter, threshold_offset, seed, density_mask
            )
        case 'ordered':
            return ordered_dither(
                image_array, threshold, BAYER_8x8, threshold_offset, density_mask, seed
            )
        case 'clustered-dot':
            return ordered_dither(
                image_array, threshold, CLUSTERED_8x8, threshold_offset, density_mask, seed
            )
        case 'bitcoin':
            return ordered_dither(
                image_array, threshold, BITCOIN_8x8, threshold_offset, density_mask, seed
            )
        case 'hal':
            return hal_dither(
                image_array, threshold, threshold_offset, seed, density_mask
            )
        case _:
            raise ValueError(f"Unknown dithering pattern: {pattern}")
