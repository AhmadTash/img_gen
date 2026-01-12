from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def create_paint_layer(
    *,
    shape: Tuple[int, int],
    mask: np.ndarray,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an off-white paint RGB + alpha (both uint8).

    Artistic intent:
    - off-white reads like cheap acrylic / correction paint
    - subtle luminance texture keeps it from feeling like a flat vector fill
    - opacity variation mimics thin/thick brush strokes
    """

    h, w = shape
    if mask.dtype != np.uint8 or mask.shape != (h, w):
        raise ValueError("mask must be uint8 and match shape")

    rng = np.random.default_rng(seed)

    # Solid white paint (no visible grain/texture inside the whited-out area).
    # Edge character is handled via the alpha edge band below.
    base_color = np.array([255, 255, 255], dtype=np.float32)  # RGB
    paint_rgb = np.tile(base_color[None, None, :], (h, w, 1))

    # Opacity: mostly solid paint.
    # We keep the INTERIOR fully opaque, but intentionally mess up the *edge band*
    # with jitter / uneven opacity / tiny gaps so it reads like a real brush.
    m = (mask > 0).astype(np.uint8)
    paint_alpha = (m * 255).astype(np.uint8)

    # Build an "edge band" mask: where the paint meets the photo.
    # This is where brush artifacts (gaps, uneven coverage) should live.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    er = cv2.erode(m, k, iterations=1)
    edge = cv2.subtract(m, er)  # 1 on edge band

    if int(edge.sum()) > 0:
        # --- Uneven opacity -----------------------------------------------------
        # Low-freq + hi-freq noise blended, then applied only on the edge.
        lf = rng.normal(0.0, 1.0, size=(max(8, h // 48), max(8, w // 48))).astype(np.float32)
        lf = cv2.resize(lf, (w, h), interpolation=cv2.INTER_CUBIC)
        lf = cv2.GaussianBlur(lf, (0, 0), sigmaX=10.0, sigmaY=10.0)
        hf = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        hf = cv2.GaussianBlur(hf, (0, 0), sigmaX=0.9, sigmaY=0.9)
        edge_noise = 0.6 * lf + 0.4 * hf
        edge_noise = (edge_noise - edge_noise.mean()) / (edge_noise.std() + 1e-6)

        # Base edge alpha starts opaque, then we carve + vary it.
        edge_alpha = 255.0 + edge_noise * 30.0  # +/- ~30

        # --- Slight gaps --------------------------------------------------------
        # Occasionally drop alpha to simulate parts where paint didn't catch.
        gap_field = rng.random((h, w), dtype=np.float32)
        gaps = (gap_field > (0.94 + 0.03 * rng.random())).astype(np.float32)  # sparse specks
        gaps = cv2.GaussianBlur(gaps, (0, 0), sigmaX=1.2, sigmaY=1.2)
        edge_alpha = edge_alpha - gaps * 120.0

        edge_alpha = np.clip(edge_alpha, 40.0, 255.0)

        # Apply to paint_alpha only in edge band; keep interior at 255.
        paint_alpha = paint_alpha.astype(np.float32)
        paint_alpha[edge > 0] = np.minimum(paint_alpha[edge > 0], edge_alpha[edge > 0])
        paint_alpha = paint_alpha.astype(np.uint8)

        # --- Jitter -------------------------------------------------------------
        # Tiny spatial jitter of just the alpha edge (1px-ish) so it doesn't look static.
        # This is subtle but helps prevent a too-perfect outline.
        dx = int(rng.integers(-1, 2))
        dy = int(rng.integers(-1, 2))
        if dx != 0 or dy != 0:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            jittered = cv2.warpAffine(paint_alpha, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            # Keep interior stable: only replace edge band.
            paint_alpha[edge > 0] = jittered[edge > 0]

    paint_rgb = np.clip(paint_rgb, 0.0, 255.0).astype(np.uint8)
    return paint_rgb, paint_alpha
