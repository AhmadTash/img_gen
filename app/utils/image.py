from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image


def decode_image_bytes_to_pil(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"invalid image bytes: {e}")


def resize_longest_edge(img: Image.Image, *, max_edge: int = 1024) -> Image.Image:
    """Resize to cap the longest edge.

    This keeps the model reasonably fast and prevents huge memory spikes.
    """

    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img

    scale = max_edge / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected RGB image")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr
