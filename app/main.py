from __future__ import annotations

import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.pipeline.run import generate_image

app = FastAPI(title="Lo-fi painted-out person generator")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    text: str = Form(...),
    paint_thickness: int = Form(15),
    messiness: float = Form(0.5),
    text_wobble: float = Form(0.5),
    seed: Optional[int] = Form(None),
) -> Response:
    if paint_thickness < 1 or paint_thickness > 200:
        raise HTTPException(status_code=400, detail="paint_thickness must be in [1, 200]")
    if not (0.0 <= messiness <= 1.0):
        raise HTTPException(status_code=400, detail="messiness must be in [0, 1]")
    if not (0.0 <= text_wobble <= 1.0):
        raise HTTPException(status_code=400, detail="text_wobble must be in [0, 1]")
    if text is None:
        raise HTTPException(status_code=400, detail="text is required")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")

    try:
        out_png = generate_image(
            raw_image_bytes=raw,
            text=text,
            paint_thickness=paint_thickness,
            messiness=messiness,
            text_wobble=text_wobble,
            seed=seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(content=out_png, media_type="image/png")
