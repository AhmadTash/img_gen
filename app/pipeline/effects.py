from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def apply_lofi_effects(
    rgb_u8: np.ndarray,
    *,
    seed: Optional[int],
    grain_mask: Optional[np.ndarray] = None,
    blur_sigma: float = 1.2,
    blur_mix: float = 0.30,
) -> np.ndarray:
    """Add lo-fi degradation: blur + mild contrast reduction + fine grain.

    Grain here is *fine* (tiny dots), and can be limited to an area via grain_mask.
    If grain_mask is provided, grain is applied only where grain_mask > 0.
    """

    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3:
        raise ValueError("apply_lofi_effects expects uint8 HxWx3")

    rng = np.random.default_rng(seed)
    x = rgb_u8.astype(np.float32)

    # Mild contrast reduction: pull towards mid-tones.
    x = 128.0 + (x - 128.0) * 0.92

    # Layer-based blur: duplicate the image, blur only one layer, then blend.
    # This feels more "optical" (cheap lens / scanning) than blurring the whole image.
    s = float(max(0.0, blur_sigma))
    mix = float(np.clip(blur_mix, 0.0, 1.0))
    if s > 0.0 and mix > 0.0:
        blurred = cv2.GaussianBlur(x, (0, 0), sigmaX=s, sigmaY=s)
        x = x * (1.0 - mix) + blurred * mix

    # Fine film grain: keep it high-frequency so it reads as tiny dots, not blobs.
    # We avoid blurring grain (or blur it extremely lightly) to keep dots small.
    grain = rng.normal(0.0, 1.0, size=rgb_u8.shape[:2]).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.15, sigmaY=0.15)
    grain_strength = 4.0  # small, subtle

    if grain_mask is not None:
        if grain_mask.dtype != np.uint8 or grain_mask.shape != rgb_u8.shape[:2]:
            raise ValueError("grain_mask must be uint8 HxW matching image")
        gm = (grain_mask.astype(np.float32) / 255.0)
    else:
        gm = 1.0

    x = x + grain[..., None] * grain_strength * gm[..., None]

    return np.clip(x, 0.0, 255.0).astype(np.uint8)
