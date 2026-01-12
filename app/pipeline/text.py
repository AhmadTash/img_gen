from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_handwritten_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Load a handwritten font from /app/fonts.

    If the provided font isn't present, Pillow will throw; we surface that clearly.
    """

    here = os.path.dirname(__file__)
    font_path = os.path.abspath(os.path.join(here, "..", "fonts", "handwritten.ttf"))
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=font_size)

    # Fallback so the service stays functional out-of-the-box.
    # This won't look as authentic as a real handwritten font, but it avoids hard failure.
    return ImageFont.load_default()


def render_handwritten_text_into_mask(
    *,
    base_rgb: np.ndarray,
    mask: np.ndarray,
    text: str,
    wobble: float,
    seed: Optional[int],
) -> np.ndarray:
    """Render user text inside the painted region only.

    Randomization is deterministic under the given seed.
    """

    if base_rgb.dtype != np.uint8 or base_rgb.ndim != 3:
        raise ValueError("base_rgb must be uint8 HxWx3")
    if mask.dtype != np.uint8 or mask.shape != base_rgb.shape[:2]:
        raise ValueError("mask must be uint8 HxW matching base")

    rng = np.random.default_rng(seed)
    w = float(np.clip(wobble, 0.0, 1.0))

    h, width = mask.shape

    # Find a bounding box of the painted region to guide placement.
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return base_rgb

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)

    # Font size: scale with painted area.
    font_size = int(np.clip(0.18 * min(box_w, box_h), 18, 120))
    font = _load_handwritten_font(font_size)

    # Create a transparent RGBA canvas for text.
    txt = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt)

    # Measure text to center it.
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=int(font_size * 0.25))
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Position jitter.
    jx = int((rng.normal(0, 1) * 0.04 * box_w) * w)
    jy = int((rng.normal(0, 1) * 0.04 * box_h) * w)

    cx = x0 + box_w // 2 + jx
    cy = y0 + box_h // 2 + jy

    x = int(np.clip(cx - tw // 2, 0, width - 1))
    y = int(np.clip(cy - th // 2, 0, h - 1))

    # Slight rotation (±2°).
    angle = float(np.clip(rng.normal(0.0, 1.0) * (2.0 * w), -2.0, 2.0))

    # Opacity variation.
    opacity = int(np.clip(200 + rng.normal(0, 1) * 25.0, 120, 255))

    # Ink color: near-black with a little warmth.
    ink = (25, 20, 18, opacity)

    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=ink,
        spacing=int(font_size * 0.25),
        align="center",
    )

    # Rotate around center for "handwritten" wobble.
    txt = txt.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, center=(cx, cy))

    txt_rgba = np.array(txt, dtype=np.uint8)  # HxWx4

    # Clip text so it appears only inside painted area.
    clip = (mask > 0).astype(np.uint8)
    txt_rgba[..., 3] = (txt_rgba[..., 3].astype(np.float32) * clip).astype(np.uint8)

    # Composite text over base.
    base = base_rgb.astype(np.float32)
    a = (txt_rgba[..., 3].astype(np.float32) / 255.0)[..., None]
    rgb = txt_rgba[..., :3].astype(np.float32)
    out = (rgb * a + base * (1.0 - a)).clip(0, 255).astype(np.uint8)
    return out
