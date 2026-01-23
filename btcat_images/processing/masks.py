import numpy as np
import numpy.typing as npt
from typing import Literal

def create_rectangle_mask(
    width: int,
    height: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> npt.NDArray[np.bool_]:
    """
    Create a rectangular mask for dithering.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        x1: Left X coordinate (fraction, can be any value)
        y1: Top Y coordinate (fraction, can be any value)
        x2: Right X coordinate (fraction, can be any value)
        y2: Bottom Y coordinate (fraction, can be any value)

    Returns:
        Boolean mask array where True indicates dithering area
    """
    mask = np.zeros((height, width), dtype=bool)

    # Convert fractions to pixel coordinates
    px1 = int(x1 * width)
    py1 = int(y1 * height)
    px2 = int(x2 * width)
    py2 = int(y2 * height)

    # Ensure coordinates are in correct order
    if px1 > px2:
        px1, px2 = px2, px1
    if py1 > py2:
        py1, py2 = py2, py1

    # Clip to image bounds
    px1 = max(0, min(px1, width))
    px2 = max(0, min(px2, width))
    py1 = max(0, min(py1, height))
    py2 = max(0, min(py2, height))

    # Fill the rectangle
    if px2 > px1 and py2 > py1:
        mask[py1:py2, px1:px2] = True

    return mask


def create_circle_mask(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float
) -> npt.NDArray[np.bool_]:
    """
    Create a circular mask for dithering.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        center_x: Center X coordinate (0.0 to 1.0)
        center_y: Center Y coordinate (0.0 to 1.0)
        radius: Radius (0.0 to 1.0, fraction of image dimensions)

    Returns:
        Boolean mask array where True indicates dithering area
    """
    # Convert fractions to pixel coordinates
    cx = center_x * width
    cy = center_y * height

    # Use average of width and height for radius calculation
    r_pixels = radius * (width + height) / 2.0

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate distances from center
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Create mask
    mask = distances <= r_pixels

    return mask


def create_gradient_mask(
    width: int,
    height: int,
    split_ratio: float,
    cut_direction: Literal['vertical', 'horizontal'],
    fade_min: float
) -> npt.NDArray[np.float64]:
    """
    Create a gradient mask for fade-out effect in cut modes.
    Legacy function.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        split_ratio: Position of the cut (0.0 to 1.0)
        cut_direction: 'vertical' or 'horizontal'
        fade_min: Minimum density at the far edge (0.0 to 1.0)

    Returns:
        Float mask array with values from fade_min to 1.0
    """
    mask = np.zeros((height, width), dtype=np.float64)

    if cut_direction == 'vertical':
        split_pos = int(width * split_ratio)
        dither_width = width - split_pos

        if dither_width > 0:
            # Create gradient from 1.0 at cut line to fade_min at right edge
            gradient = np.linspace(1.0, fade_min, dither_width)
            mask[:, split_pos:] = gradient[np.newaxis, :]
    else:  # horizontal
        split_pos = int(height * split_ratio)
        dither_height = height - split_pos

        if dither_height > 0:
            # Create gradient from 1.0 at cut line to fade_min at bottom edge
            gradient = np.linspace(1.0, fade_min, dither_height)
            mask[split_pos:, :] = gradient[:, np.newaxis]

    return mask


def create_gradient_density_mask(
    width: int,
    height: int,
    angle: float,
    density_start: float,
    density_end: float
) -> npt.NDArray[np.float64]:
    """
    Create a gradient density mask that transitions from one density to another.

    Uses angle-based gradients where the gradient transitions across the image
    based on the specified angle.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        angle: Gradient angle in degrees (0-360)
            - 0°: left to right
            - 90°: top to bottom
            - 180°: right to left
            - 270°: bottom to top
        density_start: Density at start (0.0 to 1.0)
        density_end: Density at end (0.0 to 1.0)

    Returns:
        Float mask array with values transitioning from density_start to density_end
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Create directional vector (cos, sin)
    # Note: In image coordinates, y increases downward
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Create coordinate grids (normalized to [0, 1])
    y_coords, x_coords = np.ogrid[:height, :width]
    x_norm = x_coords.astype(float) / max(width - 1, 1)
    y_norm = y_coords.astype(float) / max(height - 1, 1)

    # Project each pixel position onto the gradient direction
    # This gives us a value that increases along the gradient direction
    projection = x_norm * dx + y_norm * dy

    # Normalize projection to [0, 1] range
    # The projection range depends on the angle
    proj_min = projection.min()
    proj_max = projection.max()

    if proj_max > proj_min:
        projection_normalized = (projection - proj_min) / (proj_max - proj_min)
    else:
        projection_normalized = np.zeros_like(projection)

    # Map to [density_start, density_end]
    mask = density_start + (density_end - density_start) * projection_normalized

    return mask
