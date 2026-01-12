from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.pipeline.run import generate_image


def test_generate_image_returns_png_bytes() -> None:
    # Synthetic "photo": mid-gray with a darker rectangle so something exists.
    img = np.full((320, 480, 3), 180, dtype=np.uint8)
    img[80:260, 160:320] = 90

    pil = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")

    out = generate_image(
        raw_image_bytes=buf.getvalue(),
        text="hello",
        seed=42,
    )

    assert out[:8] == b"\x89PNG\r\n\x1a\n"
