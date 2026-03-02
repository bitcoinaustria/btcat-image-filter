import numpy as np
from PIL import Image, ImageFilter
from typing import Optional

def apply_blue_noise(
    img: Image.Image,
    strength: float = 0.0,
    seed: Optional[int] = None
) -> Image.Image:
    """Apply blue noise dithering effect.

    Blue noise is created by generating white noise and subtracting its
    low-frequency components (obtained via blurring). This produces noise
    that is evenly spaced and lacks low-frequency clusters.

    Args:
        img: Input PIL Image
        strength: Blue noise strength (0.0-1.0, default 0.0). 0 = no noise.
        seed: Random seed for reproducible noise generation.

    Returns:
        Image with blue noise effect applied
    """
    if strength <= 0.0:
        return img

    width, height = img.size
    rng = np.random.default_rng(seed)

    # Generate white noise [-1, 1]
    noise = rng.random((height, width), dtype=np.float32) * 2 - 1

    # Apply local blur to find the "neighborhood" (low frequency)
    noise_img = Image.fromarray(((noise + 1) * 127.5).astype(np.uint8), mode='L')
    blurred = np.array(noise_img.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32)
    blurred = (blurred / 127.5) - 1.0

    # Blue noise = White noise - Low frequency noise
    blue = noise - blurred

    # Scale by strength (0 to 1 maps to 0 to 255 roughly)
    blue = blue * 255 * strength

    # Add noise to original image
    img_arr = np.array(img, dtype=np.float32)

    if len(img_arr.shape) == 3:
        # Broadcasting to RGB channels
        blue = blue[..., np.newaxis]

    result = np.clip(img_arr + blue, 0, 255).astype(np.uint8)
    return Image.fromarray(result)
