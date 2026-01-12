from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def expand_and_cleanup_mask(mask_u8: np.ndarray, *, paint_thickness: int) -> np.ndarray:
    """Dilate mask and close holes.

    Artistic intent:
    - dilation creates the "painted blob" that extends past the silhouette
    - closing fills small gaps so the paint reads as a single swatch
    """

    if mask_u8.dtype != np.uint8 or mask_u8.ndim != 2:
        raise ValueError("expand_and_cleanup_mask expects uint8 HxW mask")

    px = int(max(1, paint_thickness))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    dilated = cv2.dilate(mask_u8, dil_k, iterations=1)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(7, px) | 1, max(7, px) | 1))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_k, iterations=2)

    return (closed > 0).astype(np.uint8) * 255


def _smooth_noise_field(h: int, w: int, *, seed: Optional[int], scale: float) -> np.ndarray:
    """Low-frequency noise field in [0,1].

    We avoid heavy Perlin deps by using upsampled random grids + Gaussian blur.
    This gives organic edges without looking like vector jitter.
    """

    rng = np.random.default_rng(seed)
    gh = max(2, int(round(h / max(8.0, scale))))
    gw = max(2, int(round(w / max(8.0, scale))))
    grid = rng.random((gh, gw), dtype=np.float32)
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(1.0, scale / 10.0), sigmaY=max(1.0, scale / 10.0))
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    return field


def generate_irregular_painted_mask(mask_u8: np.ndarray, *, messiness: float, seed: Optional[int]) -> np.ndarray:
    """Add hand-painted irregularity.

    Techniques used (in combination):
    - noise-driven edge threshold (like paint bleeding)
    - random erosion/dilation cycles (like uneven brush pressure)
    """

    if mask_u8.dtype != np.uint8 or mask_u8.ndim != 2:
        raise ValueError("generate_irregular_painted_mask expects uint8 HxW mask")

    m = float(np.clip(messiness, 0.0, 1.0))
    base = (mask_u8 > 0).astype(np.uint8) * 255

    h, w = base.shape
    field = _smooth_noise_field(h, w, seed=seed, scale=30.0 + 40.0 * m)

    # Edge band: distance transform gives us where edges are, so we only roughen boundary.
    inv = (base == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    dist = dist / (dist.max() + 1e-6)

    # Stronger irregularity near the boundary.
    edge_weight = np.clip(1.0 - dist * 3.0, 0.0, 1.0)

    # Push/pull boundary by mixing in noise.
    # Turn it up a bit so it reads more like a messed-up brush edge.
    jitter = (field - 0.5) * (0.70 * m) * edge_weight

    # Add a higher-frequency component so the edge isn't just "wobbly" but also a bit ragged.
    # This mimics bristle marks / tiny paint chips at the boundary.
    hf = _smooth_noise_field(h, w, seed=None if seed is None else int(seed) + 1337, scale=10.0 + 10.0 * m)
    jitter += (hf - 0.5) * (0.25 * m) * edge_weight

    # Convert to a soft mask then re-threshold.
    soft = (base.astype(np.float32) / 255.0) + jitter
    soft = np.clip(soft, 0.0, 1.0)

    thr = 0.5
    irregular = (soft > thr).astype(np.uint8) * 255

    # Random morphological cycles (few steps so it stays blob-like).
    rng = np.random.default_rng(seed)
    cycles = int(round(2 + 5 * m))
    for _ in range(cycles):
        ksz = int((3 + rng.integers(0, 6)) | 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        if rng.random() < 0.5:
            irregular = cv2.erode(irregular, k, iterations=1)
        else:
            irregular = cv2.dilate(irregular, k, iterations=1)

    # A final light edge "chipping": erode then dilate with a tiny kernel, but only once.
    # This makes the silhouette less smooth without destroying the blob.
    chip_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    irregular = cv2.erode(irregular, chip_k, iterations=1)
    irregular = cv2.dilate(irregular, chip_k, iterations=1)

    # Final close to avoid accidental pinholes.
    irregular = cv2.morphologyEx(
        irregular,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    # --- Brush-stroke edge pass -------------------------------------------------
    # The noise/erode/dilate approach can look like "organic blob".
    # To get *visible brush strokes along edges*, we stamp small, rotated
    # elliptical "dabs" along the contour. This reads as bristles / uneven paint
    # application without requiring actual stroke simulation.
    if m > 0.0:
        rng = np.random.default_rng(seed)
        contours, _ = cv2.findContours((irregular > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        stroke_field = np.zeros_like(irregular)

        # Number of dabs scales with perimeter and messiness.
        for cnt in contours:
            if len(cnt) < 20:
                continue

            perim = cv2.arcLength(cnt, closed=True)
            n_dabs = int(np.clip((perim / 18.0) * (0.6 + 1.9 * m), 40, 2200))
            idxs = rng.integers(0, len(cnt), size=n_dabs)

            for idx in idxs:
                x, y = cnt[idx, 0]

                # Dab sizes: a bit wider/taller for higher messiness.
                major = float(rng.uniform(6.0, 18.0) * (0.6 + 1.2 * m))
                minor = float(rng.uniform(2.0, 7.0) * (0.6 + 1.2 * m))
                angle = float(rng.uniform(0.0, 180.0))

                # Offset slightly outward/inward so strokes sit on the boundary.
                ox = int(rng.normal(0.0, 1.0) * (0.8 + 1.8 * m))
                oy = int(rng.normal(0.0, 1.0) * (0.8 + 1.8 * m))

                center = (int(x + ox), int(y + oy))
                axes = (max(1, int(major / 2.0)), max(1, int(minor / 2.0)))
                cv2.ellipse(stroke_field, center, axes, angle, 0, 360, 255, thickness=-1)

        # Feather slightly so strokes don't look like crisp stamps.
        stroke_field = cv2.GaussianBlur(stroke_field, (0, 0), sigmaX=0.7 + 0.8 * m, sigmaY=0.7 + 0.8 * m)
        stroke_field = (stroke_field > 18).astype(np.uint8) * 255

        # Mix strokes with the existing irregular mask.
        # Union adds outward bristle strokes; intersection via erode balances it.
        irregular = cv2.bitwise_or(irregular, stroke_field)
        if m > 0.35:
            irregular = cv2.morphologyEx(
                irregular,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )

    return (irregular > 0).astype(np.uint8) * 255
