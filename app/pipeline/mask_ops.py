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

    Current approach aims for a more "brushy / frayed" boundary than a generic
    organic blob:
    - keep a stable, filled-in base region so the paint reads as a single swatch
    - generate a *boundary band* and only stylize that band
    - add outward/inward chips (missing paint) + bristle streaks (directional dabs)
    """

    if mask_u8.dtype != np.uint8 or mask_u8.ndim != 2:
        raise ValueError("generate_irregular_painted_mask expects uint8 HxW mask")

    m = float(np.clip(messiness, 0.0, 1.0))
    base = (mask_u8 > 0).astype(np.uint8)
    h, w = base.shape
    rng = np.random.default_rng(seed)

    # --- 1) Stable blob --------------------------------------------------------
    # Keep the region cohesive: slightly close + smooth corners.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob = cv2.morphologyEx(base, cv2.MORPH_CLOSE, close_k, iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # --- 2) Boundary band ------------------------------------------------------
    # We only stylize this band so the interior remains solid and readable.
    band_px = int(np.clip(6 + 26 * m, 4, 32))
    dil = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1)), iterations=1)
    er = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1)), iterations=1)
    band = ((dil > 0) & (er == 0)).astype(np.uint8)

    # --- 3) Chips (missing paint) ---------------------------------------------
    # Use thresholded noise + a couple of small morph ops to create paint "bites".
    chips = np.zeros((h, w), dtype=np.uint8)
    if m > 0.0:
        lf = _smooth_noise_field(h, w, seed=seed, scale=18.0 + 30.0 * m)
        hf = _smooth_noise_field(h, w, seed=None if seed is None else int(seed) + 7331, scale=6.0 + 10.0 * m)
        field = (0.65 * lf + 0.35 * hf)

        # More messiness => more chips.
        thr = 0.86 - 0.22 * m
        chips = ((field > thr) & (band > 0)).astype(np.uint8) * 255
        chips = cv2.GaussianBlur(chips, (0, 0), sigmaX=0.8 + 0.8 * m, sigmaY=0.8 + 0.8 * m)
        chips = (chips > 32).astype(np.uint8) * 255
        chips = cv2.dilate(chips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # --- 4) Bristle streaks (directional dabs) --------------------------------
    # Stamp thin ellipses along the contour to get "frayed" brisable edges.
    streaks = np.zeros((h, w), dtype=np.uint8)
    if m > 0.0:
        contours, _ = cv2.findContours((blob > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) < 30:
                continue
            perim = cv2.arcLength(cnt, closed=True)
            n = int(np.clip((perim / 24.0) * (0.4 + 1.8 * m), 30, 1500))
            idxs = rng.integers(0, len(cnt), size=n)
            for idx in idxs:
                x, y = cnt[idx, 0]

                # Tangent direction from neighbors.
                p0 = cnt[(idx - 3) % len(cnt), 0].astype(np.float32)
                p1 = cnt[(idx + 3) % len(cnt), 0].astype(np.float32)
                v = p1 - p0
                ang = float(np.degrees(np.arctan2(v[1], v[0])))

                # Long thin stroke oriented with tangent.
                major = float(rng.uniform(8.0, 26.0) * (0.5 + 1.1 * m))
                minor = float(rng.uniform(1.5, 4.5) * (0.6 + 1.0 * m))

                # Push slightly outwards/inwards by random small amount.
                ox = int(rng.normal(0.0, 1.2) * (0.8 + 1.8 * m))
                oy = int(rng.normal(0.0, 1.2) * (0.8 + 1.8 * m))
                center = (int(x + ox), int(y + oy))
                axes = (max(1, int(major / 2.0)), max(1, int(minor / 2.0)))
                cv2.ellipse(streaks, center, axes, ang, 0, 360, 255, thickness=-1)

        # Keep streaks mainly in the band so it doesn't inflate the whole blob.
        streaks = cv2.bitwise_and(streaks, band * 255)
        streaks = cv2.GaussianBlur(streaks, (0, 0), sigmaX=0.7 + 0.9 * m, sigmaY=0.7 + 0.9 * m)
        streaks = (streaks > 28).astype(np.uint8) * 255

    # --- 5) Combine ------------------------------------------------------------
    out = (blob * 255).astype(np.uint8)
    if int(streaks.sum()) > 0:
        out = cv2.bitwise_or(out, streaks)
    if int(chips.sum()) > 0:
        # Subtract chips from the band only (keeps interior filled).
        out_band = cv2.bitwise_and(out, (band * 255))
        out_band = cv2.subtract(out_band, chips)
        out = cv2.bitwise_or(cv2.bitwise_and(out, (1 - band) * 255), out_band)

    # Final close to avoid accidental tiny pinholes.
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return (out > 0).astype(np.uint8) * 255
