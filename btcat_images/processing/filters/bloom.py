import numpy as np
from PIL import Image, ImageFilter, ImageChops


def apply_bloom(
    img: Image.Image,
    intensity: float = 0.5,
    radius: float = 75.0,
) -> Image.Image:
    """Apply bloom post-processing effect using screen blending.

    Creates a soft glow by blurring the image and screen-blending it back
    at reduced opacity. Matches the JS canvas implementation.

    Args:
        img: Input PIL Image (RGB)
        intensity: Bloom strength (0.0-1.0, default 0.5). 0 = no bloom.
        radius: Blur radius control (1.0-150.0, default 75.0).
                Actual Gaussian radius is radius/5.

    Returns:
        Image with bloom effect applied
    """
    if intensity <= 0.0:
        return img

    # Blur the image
    blur_radius = radius / 5.0
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Screen blend: result = 1 - (1-a)*(1-b)
    # PIL's ImageChops.screen does exactly this
    screened = ImageChops.screen(img, blurred)

    # Blend at intensity/2 opacity (matching JS: globalAlpha = bloomIntensity/200)
    alpha = intensity / 2.0
    result = Image.blend(img, screened, alpha)

    return result
