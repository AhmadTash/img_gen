from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.pipeline.effects import apply_lofi_effects
from app.pipeline.mask_ops import (
    expand_and_cleanup_mask,
    generate_irregular_painted_mask,
)
from app.pipeline.paint import create_paint_layer
from app.pipeline.segment import segment_person_mask
from app.pipeline.text import render_handwritten_text_into_mask
from app.utils.image import (
    decode_image_bytes_to_pil,
    pil_to_numpy_rgb,
    resize_longest_edge,
)


@dataclass(frozen=True)
class GenerateParams:
    paint_thickness: int = 15
    messiness: float = 0.5
    text_wobble: float = 0.5
    seed: Optional[int] = None


def generate_image(
    *,
    raw_image_bytes: bytes,
    text: str,
    paint_thickness: int = 15,
    messiness: float = 0.5,
    text_wobble: float = 0.5,
    seed: Optional[int] = None,
) -> bytes:
    """End-to-end pipeline.

    Contract:
    - Input: raw bytes of an image + text
    - Output: PNG bytes
    - Determinism: if seed is not None, output is deterministic for same inputs
    """

    if text.strip() == "":
        raise ValueError("text must be non-empty")

    params = GenerateParams(
        paint_thickness=paint_thickness,
        messiness=float(messiness),
        text_wobble=float(text_wobble),
        seed=seed,
    )

    # a) Load & Normalize
    pil = decode_image_bytes_to_pil(raw_image_bytes).convert("RGB")
    pil = resize_longest_edge(pil, max_edge=1024)
    rgb = pil_to_numpy_rgb(pil)  # uint8 HxWx3

    # b) Person Segmentation
    person_mask = segment_person_mask(rgb, seed=params.seed)

    # c) Mask Expansion & Cleanup
    expanded = expand_and_cleanup_mask(person_mask, paint_thickness=params.paint_thickness)

    # d) Irregular Painted Mask Generation
    irregular = generate_irregular_painted_mask(expanded, messiness=params.messiness, seed=params.seed)

    # e) Paint Layer Creation
    paint_rgb, paint_alpha = create_paint_layer(
        shape=rgb.shape[:2],
        mask=irregular,
        seed=params.seed,
    )

    # Optional depth: a subtle shadow around the paint edges.
    # This helps the white patch feel like it's sitting *on top* of the photo.
    shadow_rgb, shadow_alpha = create_paint_shadow(irregular, seed=params.seed)

    # f) Composite Paint Over Original
    comp = composite_over(rgb, shadow_rgb, shadow_alpha)
    comp = composite_over(comp, paint_rgb, paint_alpha)

    # g) Handwritten Text Rendering (clipped inside paint)
    comp = render_handwritten_text_into_mask(
        base_rgb=comp,
        mask=irregular,
        text=text,
        wobble=params.text_wobble,
        seed=params.seed,
    )

    # h) Lo-Fi Degradation
    # Apply fine grain only to the non-painted area so the white patch stays clean.
    outside_paint = (255 - irregular).astype(np.uint8)
    comp = apply_lofi_effects(comp, seed=params.seed, grain_mask=outside_paint)

    # i) Export PNG
    out = Image.fromarray(comp, mode="RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def composite_over(base_rgb: np.ndarray, overlay_rgb: np.ndarray, overlay_alpha: np.ndarray) -> np.ndarray:
    """Alpha composite overlay over base.

    overlay_alpha: uint8 HxW in [0,255]
    """

    if base_rgb.dtype != np.uint8 or overlay_rgb.dtype != np.uint8 or overlay_alpha.dtype != np.uint8:
        raise ValueError("composite expects uint8 arrays")

    a = (overlay_alpha.astype(np.float32) / 255.0)[..., None]
    out = (overlay_rgb.astype(np.float32) * a + base_rgb.astype(np.float32) * (1.0 - a)).clip(0, 255)
    return out.astype(np.uint8)


def create_paint_shadow(mask_u8: np.ndarray, *, seed: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """Create a soft shadow around the painted region.

    Implementation:
    - build an edge band from the mask
    - blur it to a soft halo
    - offset slightly for a hand-made "sticker" feel
    """

    if mask_u8.dtype != np.uint8 or mask_u8.ndim != 2:
        raise ValueError("create_paint_shadow expects uint8 HxW mask")

    h, w = mask_u8.shape
    m = (mask_u8 > 0).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    er = cv2.erode(m, k, iterations=1)
    edge = cv2.subtract(m, er).astype(np.float32)  # 0/1

    # Convert to a soft shadow alpha.
    shadow = cv2.GaussianBlur(edge, (0, 0), sigmaX=6.0, sigmaY=6.0)
    shadow = shadow / (shadow.max() + 1e-6)

    # Shadow opacity (subtle): max around ~18%.
    alpha = (shadow * 46.0).clip(0.0, 46.0).astype(np.uint8)

    # Slight offset for realism.
    rng = np.random.default_rng(seed)
    dx = int(rng.integers(-1, 3))
    dy = int(rng.integers(0, 3))
    if dx != 0 or dy != 0:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        alpha = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    shadow_rgb = np.zeros((h, w, 3), dtype=np.uint8)  # black shadow
    return shadow_rgb, alpha
