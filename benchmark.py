import time
import numpy as np
from btcat_images.processing.dither.original import _floyd_steinberg_3color
from btcat_images.constants import ORIGINAL_PALETTE

# Create a sample image 1920x1080
width = 1920
height = 1080

pixels_flat = np.random.rand(width * height, 4).astype(np.float32) * 255.0
pixels_flat[:, 3] = 255.0 # alpha

palette_array = np.array(ORIGINAL_PALETTE, dtype=np.float32)

# Warmup to compile JIT
pixels_copy = pixels_flat.copy()
_floyd_steinberg_3color(pixels_copy, width, height, palette_array)

print("Starting benchmark...")
start = time.perf_counter()
for _ in range(5):
    pixels_copy = pixels_flat.copy()
    _floyd_steinberg_3color(pixels_copy, width, height, palette_array)
end = time.perf_counter()

print(f"Time taken for 5 iterations: {end - start:.4f} seconds")
print(f"Average time per iteration: {(end - start) / 5:.4f} seconds")
