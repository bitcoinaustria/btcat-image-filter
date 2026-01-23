from typing import Literal, Tuple
import numpy as np

# Brand definitions
BRANDS = {
    'btcat': {'color': (227, 0, 15), 'type': 'monochrome'},       # #E3000F
    'lightning': {'color': (245, 155, 31), 'type': 'monochrome'}, # #F59B1F
    'cypherpunk': {'color': (0, 255, 65), 'type': 'monochrome'},  # #00FF41
    'rgb': {'type': 'rgb'}
}

# Dark background color for dithered areas
DARK_BACKGROUND: Tuple[int, int, int] = (34, 34, 34)  # #222222

# Golden ratio (phi)
GOLDEN_RATIO: float = 1.618033988749895

# Dithering patterns
DitherPattern = Literal['floyd-steinberg', 'ordered', 'atkinson', 'clustered-dot', 'bitcoin', 'hal']

# Matrices
# Bayer 8x8 matrix
BAYER_8x8 = np.array([
    [ 0, 48, 12, 60,  3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [ 8, 56,  4, 52, 11, 59,  7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [ 2, 50, 14, 62,  1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58,  6, 54,  9, 57,  5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21]
], dtype=float) / 64.0

# Clustered dot (Spiral 8x8)
CLUSTERED_8x8 = np.array([
    [24, 10, 12, 26, 35, 47, 49, 37],
    [ 8,  0,  2, 14, 45, 59, 61, 51],
    [22,  6,  4, 16, 43, 57, 63, 53],
    [30, 20, 18, 28, 33, 41, 55, 39],
    [36, 50, 48, 34, 27, 13, 11, 25],
    [52, 62, 60, 46, 15,  3,  1,  9],
    [54, 64, 58, 44, 17,  5,  7, 23],
    [40, 56, 42, 32, 29, 19, 21, 31]
], dtype=float) / 65.0

# Bitcoin custom pattern (8x8)
# Creates a "B" shape for Bitcoin branding with two humps
BITCOIN_8x8 = np.array([
    [10, 60, 60, 60, 60, 10, 10, 10],
    [10, 60, 10, 10, 10, 60, 10, 10],
    [10, 60, 10, 10, 10, 60, 10, 10],
    [10, 60, 60, 60, 60, 10, 10, 10],
    [10, 60, 10, 10, 10, 60, 10, 10],
    [10, 60, 10, 10, 10, 60, 10, 10],
    [10, 60, 60, 60, 60, 10, 10, 10],
    [10, 10, 10, 10, 10, 10, 10, 10],
], dtype=float) / 64.0
