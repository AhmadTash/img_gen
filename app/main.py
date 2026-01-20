from __future__ import annotations

import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline.run import generate_image

app = FastAPI(title="Lo-fi painted-out person generator")

# Basic CORS for local dev UI.
# In production, tighten origins to your deployed frontend URL(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    text: str = Form(...),
    paint_thickness: int = Form(10),
    messiness: float = Form(0.01),
    text_wobble: float = Form(0.1),
    blur_sigma: float = Form(1.5),
    blur_mix: float = Form(0.5),
    shadow_opacity: float = Form(0.18),
    shadow_sigma: float = Form(6.0),
    shadow_dx: int = Form(1),
    shadow_dy: int = Form(1),
    seed: Optional[int] = Form(1500),
) -> Response:
    if paint_thickness < 1 or paint_thickness > 200:
        raise HTTPException(status_code=400, detail="paint_thickness must be in [1, 200]")
    if not (0.0 <= messiness <= 1.0):
        raise HTTPException(status_code=400, detail="messiness must be in [0, 1]")
    if not (0.0 <= text_wobble <= 1.0):
        raise HTTPException(status_code=400, detail="text_wobble must be in [0, 1]")
    if blur_sigma < 0.0 or blur_sigma > 10.0:
        raise HTTPException(status_code=400, detail="blur_sigma must be in [0, 10]")
    if not (0.0 <= blur_mix <= 1.0):
        raise HTTPException(status_code=400, detail="blur_mix must be in [0, 1]")
    if not (0.0 <= shadow_opacity <= 1.0):
        raise HTTPException(status_code=400, detail="shadow_opacity must be in [0, 1]")
    if shadow_sigma < 0.0 or shadow_sigma > 50.0:
        raise HTTPException(status_code=400, detail="shadow_sigma must be in [0, 50]")
    if shadow_dx < -50 or shadow_dx > 50:
        raise HTTPException(status_code=400, detail="shadow_dx must be in [-50, 50]")
    if shadow_dy < -50 or shadow_dy > 50:
        raise HTTPException(status_code=400, detail="shadow_dy must be in [-50, 50]")
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
            blur_sigma=blur_sigma,
            blur_mix=blur_mix,
            shadow_opacity=shadow_opacity,
            shadow_sigma=shadow_sigma,
            shadow_dx=shadow_dx,
            shadow_dy=shadow_dy,
            seed=seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(content=out_png, media_type="image/png")
