from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def create_paint_layer(
    *,
    shape: Tuple[int, int],
    mask: np.ndarray,
    seed: Optional[int],
    edge_softness: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an off-white paint RGB + alpha (both uint8).

    Artistic intent:
    - off-white reads like cheap acrylic / correction paint
    - subtle luminance texture keeps it from feeling like a flat vector fill
    - opacity variation mimics thin/thick brush strokes
    - smooth, anti-aliased edges that don't look pixelated

    Args:
        edge_softness: Controls how much edge feathering to apply (0-10, default 3)
    """

    h, w = shape
    if mask.dtype != np.uint8 or mask.shape != (h, w):
        raise ValueError("mask must be uint8 and match shape")

    rng = np.random.default_rng(seed)
    edge_softness = float(np.clip(edge_softness, 0.0, 10.0))

    # Solid white paint (no visible grain/texture inside the whited-out area).
    base_color = np.array([255, 255, 255], dtype=np.float32)  # RGB
    paint_rgb = np.tile(base_color[None, None, :], (h, w, 1))

    m = (mask > 0).astype(np.uint8)

    # --- SMOOTH ALPHA GRADIENT using distance transform ---
    # This creates a natural falloff at edges instead of hard binary cutoff.
    # Distance from edge (positive inside, negative outside conceptually)
    dist_inside = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)

    # Feather width scales with edge_softness (0 = no feather, 10 = wide feather)
    feather_width = max(1.0, edge_softness * 2.0)

    # Create smooth alpha: fully opaque inside, gradient falloff at edges
    # Using sigmoid-like smooth transition instead of linear
    alpha_gradient = dist_inside / (feather_width + 1e-6)
    alpha_gradient = np.clip(alpha_gradient, 0.0, 1.0)

    # Apply smooth step (Hermite interpolation) for even smoother falloff
    # smoothstep(x) = 3x² - 2x³, gives S-curve transition
    alpha_gradient = alpha_gradient * alpha_gradient * (3.0 - 2.0 * alpha_gradient)

    paint_alpha = (alpha_gradient * 255.0).astype(np.float32)

    # --- ADAPTIVE EDGE BAND ---
    # Scale structuring element based on image size for consistent look
    min_dim = min(h, w)
    edge_kernel_size = max(5, min(21, int(min_dim * 0.015) | 1))  # Odd number, adaptive
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_kernel_size, edge_kernel_size))
    er = cv2.erode(m, k, iterations=1)
    edge = cv2.subtract(m, er)  # 1 on edge band

    if int(edge.sum()) > 0:
        # --- Uneven opacity for artistic brushy effect ---
        lf = rng.normal(0.0, 1.0, size=(max(8, h // 48), max(8, w // 48))).astype(np.float32)
        lf = cv2.resize(lf, (w, h), interpolation=cv2.INTER_CUBIC)
        lf = cv2.GaussianBlur(lf, (0, 0), sigmaX=10.0, sigmaY=10.0)
        hf = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        hf = cv2.GaussianBlur(hf, (0, 0), sigmaX=0.9, sigmaY=0.9)
        edge_noise = 0.6 * lf + 0.4 * hf
        edge_noise = (edge_noise - edge_noise.mean()) / (edge_noise.std() + 1e-6)

        # Modulate edge alpha with noise (subtle variation)
        edge_mod = edge_noise * 25.0  # +/- ~25
        paint_alpha[edge > 0] = np.clip(paint_alpha[edge > 0] + edge_mod[edge > 0], 40.0, 255.0)

        # --- Slight gaps (sparse, for painterly effect) ---
        gap_field = rng.random((h, w), dtype=np.float32)
        gaps = (gap_field > (0.96 + 0.02 * rng.random())).astype(np.float32)  # Very sparse
        gaps = cv2.GaussianBlur(gaps, (0, 0), sigmaX=1.5, sigmaY=1.5)
        paint_alpha[edge > 0] = np.clip(paint_alpha[edge > 0] - gaps[edge > 0] * 80.0, 40.0, 255.0)

        # --- Subtle jitter (smoother with LINEAR interpolation) ---
        dx = int(rng.integers(-1, 2))
        dy = int(rng.integers(-1, 2))
        if dx != 0 or dy != 0:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            jittered = cv2.warpAffine(
                paint_alpha.astype(np.uint8), M, (w, h),
                flags=cv2.INTER_LINEAR,  # Smoother than NEAREST
                borderValue=0
            )
            edge_blend = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
            paint_alpha = paint_alpha * (1.0 - edge_blend) + jittered.astype(np.float32) * edge_blend

    # --- FINAL GAUSSIAN FEATHERING for anti-aliasing ---
    # Apply subtle blur to the alpha to eliminate any remaining jaggies
    feather_sigma = max(0.5, edge_softness * 0.3)
    paint_alpha = cv2.GaussianBlur(paint_alpha.astype(np.float32), (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)

    paint_rgb = np.clip(paint_rgb, 0.0, 255.0).astype(np.uint8)
    paint_alpha = np.clip(paint_alpha, 0.0, 255.0).astype(np.uint8)
    return paint_rgb, paint_alpha
