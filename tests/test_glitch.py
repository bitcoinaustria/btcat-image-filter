
import sys
from pathlib import Path

# Add project root to path so we can import main
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pytest
from main import glitch_swap_rows

def test_glitch_swap_rows_deterministic():
    """Test that glitch_swap_rows produces deterministic output with seed."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    res1 = glitch_swap_rows(img, 1.0, seed=42)
    res2 = glitch_swap_rows(img, 1.0, seed=42)

    assert np.array_equal(res1, res2)

def test_glitch_swap_rows_changed():
    """Test that glitch_swap_rows actually modifies the image."""
    img = np.arange(100*100*3).reshape((100, 100, 3)).astype(np.uint8)
    # Using a gradient or pattern ensures changes are visible

    res = glitch_swap_rows(img, 1.0, seed=42)

    assert not np.array_equal(img, res)
    # Check shape
    assert res.shape == img.shape

def test_glitch_swap_rows_grayscale():
    """Test with 2D array."""
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    res = glitch_swap_rows(img, 1.0, seed=42)
    assert res.shape == img.shape
    assert not np.array_equal(img, res)

def test_glitch_swap_rows_intensity_zero():
    """Test that intensity 0.0 produces no changes."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    res = glitch_swap_rows(img, 0.0, seed=42)
    assert np.array_equal(img, res)
