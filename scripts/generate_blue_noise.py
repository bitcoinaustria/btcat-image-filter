#!/usr/bin/env python3
"""Generate a blue noise threshold matrix using the void-and-cluster algorithm.

This script generates a tileable blue noise threshold matrix and prints it
as a Python/NumPy array literal suitable for pasting into constants.py.

Algorithm reference:
  Ulichney, "The void-and-cluster method for dither array generation" (1993)
  https://blog.demofox.org/2019/06/25/generating-blue-noise-textures-with-void-and-cluster/

Usage:
  python scripts/generate_blue_noise.py [--size 64] [--sigma 1.9]
"""

import argparse
import numpy as np


def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Create a 1D Gaussian kernel."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()


def toroidal_gaussian_energy(binary: np.ndarray, sigma: float) -> np.ndarray:
    """Compute energy of each pixel using separable Gaussian blur with wrap boundaries.

    Uses FFT-based convolution for efficiency on larger matrices.
    """
    size = binary.shape[0]
    img = binary.astype(np.float64)

    # Build 2D Gaussian kernel in frequency domain for wrap convolution
    radius = int(np.ceil(3 * sigma))
    k1d = _gaussian_kernel_1d(sigma, radius)

    # Pad kernel to image size for FFT
    kernel_2d = np.outer(k1d, k1d)
    padded_kernel = np.zeros((size, size), dtype=np.float64)
    ky, kx = kernel_2d.shape
    # Place kernel centered at (0,0) with wrap
    for dy in range(ky):
        for dx in range(kx):
            py = (dy - ky // 2) % size
            px = (dx - kx // 2) % size
            padded_kernel[py, px] = kernel_2d[dy, dx]

    # FFT convolution (automatically handles wrap/periodic boundaries)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(padded_kernel)))


def generate_blue_noise(size: int = 64, sigma: float = 1.9, seed: int = 42) -> np.ndarray:
    """Generate a blue noise threshold matrix using void-and-cluster.

    Returns an integer array of shape (size, size) with values 0..size*size-1,
    each value appearing exactly once.
    """
    rng = np.random.default_rng(seed)
    total = size * size

    # Phase 0: Generate initial binary pattern (~10% density)
    # Start with random sparse points, then refine
    initial_density = max(1, total // 10)
    binary = np.zeros((size, size), dtype=bool)
    indices = rng.choice(total, size=initial_density, replace=False)
    binary.flat[indices] = True

    # Refine initial pattern: remove tightest clusters, add to largest voids
    for _ in range(initial_density * 2):
        energy = toroidal_gaussian_energy(binary, sigma)

        # Remove tightest cluster (highest energy among True pixels)
        cluster_energy = np.where(binary, energy, -np.inf)
        tightest = np.unravel_index(np.argmax(cluster_energy), (size, size))
        binary[tightest] = False

        # Add to largest void (lowest energy among False pixels)
        energy = toroidal_gaussian_energy(binary, sigma)
        void_energy = np.where(~binary, energy, np.inf)
        largest_void = np.unravel_index(np.argmin(void_energy), (size, size))
        binary[largest_void] = True

    # The rank matrix: each pixel gets a unique rank 0..total-1
    ranks = np.zeros((size, size), dtype=int)
    n_ones = int(binary.sum())

    # Phase 1: Rank existing points by removing tightest clusters
    # Points removed first get lower ranks
    phase1_binary = binary.copy()
    for rank in range(n_ones - 1, -1, -1):
        energy = toroidal_gaussian_energy(phase1_binary, sigma)
        cluster_energy = np.where(phase1_binary, energy, -np.inf)
        tightest = np.unravel_index(np.argmax(cluster_energy), (size, size))
        phase1_binary[tightest] = False
        ranks[tightest] = rank

    # Phase 2: Fill from n_ones to total//2 by finding largest voids
    phase2_binary = binary.copy()
    for rank in range(n_ones, total // 2):
        energy = toroidal_gaussian_energy(phase2_binary, sigma)
        void_energy = np.where(~phase2_binary, energy, np.inf)
        largest_void = np.unravel_index(np.argmin(void_energy), (size, size))
        phase2_binary[largest_void] = True
        ranks[largest_void] = rank

    # Phase 3: Fill remaining by inverting logic (find tightest cluster in complement)
    phase3_binary = ~phase2_binary  # Start with unfilled pixels as "ones"
    for rank in range(total // 2, total):
        energy = toroidal_gaussian_energy(phase3_binary, sigma)
        cluster_energy = np.where(phase3_binary, energy, -np.inf)
        tightest = np.unravel_index(np.argmax(cluster_energy), (size, size))
        phase3_binary[tightest] = False
        ranks[tightest] = rank

    return ranks


def format_matrix(ranks: np.ndarray) -> str:
    """Format the rank matrix as a Python array literal for constants.py."""
    size = ranks.shape[0]
    total = size * size
    lines = []
    lines.append(f"BLUE_NOISE_{size}x{size} = np.array([")
    for row in range(size):
        values = ", ".join(f"{int(v):4d}" for v in ranks[row])
        comma = "," if row < size - 1 else ""
        lines.append(f"    [{values}]{comma}")
    lines.append(f"], dtype=float).reshape({size}, {size}) / {total}.0")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate blue noise threshold matrix")
    parser.add_argument("--size", type=int, default=64, help="Matrix size (default: 64)")
    parser.add_argument("--sigma", type=float, default=1.9, help="Gaussian sigma (default: 1.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Generating {args.size}x{args.size} blue noise matrix (sigma={args.sigma})...", flush=True)
    ranks = generate_blue_noise(args.size, args.sigma, args.seed)
    print(format_matrix(ranks))
    print(f"\n# Matrix contains values 0..{args.size*args.size - 1}, each exactly once.")
    print(f"# Normalized range: 0.0 .. {(args.size*args.size - 1) / (args.size*args.size):.6f}")
