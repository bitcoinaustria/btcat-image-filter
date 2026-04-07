import numpy as np
import numpy.typing as npt
from typing import Literal
from numba import njit, prange

@njit(parallel=True, cache=True)
def _create_circle_mask_jit(width: int, height: int, cx: float, cy: float, r_pixels_sq: float) -> npt.NDArray[np.bool_]:
    mask = np.zeros((height, width), dtype=np.bool_)
    for y in prange(height):
        dy_sq = (y - cy) ** 2
        for x in range(width):
            dist_sq = (x - cx) ** 2 + dy_sq
            if dist_sq <= r_pixels_sq:
                mask[y, x] = True
    return mask

@njit(parallel=True, cache=True)
def _apply_circles_mask_jit(mask: npt.NDArray[np.bool_], width: int, height: int, circles_data: npt.NDArray[np.float64]) -> None:
    num_circles = circles_data.shape[0]
    for y in prange(height):
        for x in range(width):
            if not mask[y, x]:
                for i in range(num_circles):
                    cx = circles_data[i, 0]
                    cy = circles_data[i, 1]
                    r_sq = circles_data[i, 2]
                    if (x - cx)**2 + (y - cy)**2 <= r_sq:
                        mask[y, x] = True
                        break

@njit(parallel=True, cache=True)
def _create_gradient_density_mask_jit(
    width: int, 
    height: int, 
    dx: float, 
    dy: float, 
    density_start: float, 
    density_end: float
) -> npt.NDArray[np.float64]:
    # Theoretical min/max projection based on unit square corners
    # (0,0), (1,0), (0,1), (1,1)
    c1 = 0.0 * dx + 0.0 * dy
    c2 = 1.0 * dx + 0.0 * dy
    c3 = 0.0 * dx + 1.0 * dy
    c4 = 1.0 * dx + 1.0 * dy
    
    proj_min = min(c1, min(c2, min(c3, c4)))
    proj_max = max(c1, max(c2, max(c3, c4)))
    
    proj_range = proj_max - proj_min if proj_max > proj_min else 1.0
    density_diff = density_end - density_start
    
    w_inv = 1.0 / max(width - 1, 1)
    h_inv = 1.0 / max(height - 1, 1)
    
    mask = np.empty((height, width), dtype=np.float64)
    for y in prange(height):
        yn = y * h_inv
        for x in range(width):
            xn = x * w_inv
            proj = xn * dx + yn * dy
            proj_norm = (proj - proj_min) / proj_range
            mask[y, x] = density_start + density_diff * proj_norm
            
    return mask

def apply_rectangle_to_mask(
    mask: npt.NDArray[np.bool_],
    width: int,
    height: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> None:
    """
    Apply a rectangular mask in-place to an existing boolean mask array.
    """
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
    apply_rectangle_to_mask(mask, width, height, x1, y1, x2, y2)
    return mask


def apply_circles_to_mask(
    mask: npt.NDArray[np.bool_],
    width: int,
    height: int,
    circles: list[tuple[float, float, float]]
) -> None:
    """
    Apply multiple circular masks in-place to an existing boolean mask array.
    """
    if not circles:
        return

    circles_data = np.zeros((len(circles), 3), dtype=np.float64)
    for i, (center_x, center_y, radius) in enumerate(circles):
        cx = center_x * width
        cy = center_y * height
        r_pixels = radius * (width + height) / 2.0
        circles_data[i, 0] = cx
        circles_data[i, 1] = cy
        circles_data[i, 2] = r_pixels ** 2

    _apply_circles_mask_jit(mask, width, height, circles_data)


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
    r_pixels_sq = r_pixels ** 2

    return _create_circle_mask_jit(width, height, cx, cy, r_pixels_sq)


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

    return _create_gradient_density_mask_jit(width, height, dx, dy, density_start, density_end)
