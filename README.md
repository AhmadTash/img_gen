# Lo-fi painted-out person generator (FastAPI)

Backend image-processing pipeline that:

1. segments a person,
2. paints them out with an off-white textured layer,
3. renders user-provided handwritten text only inside that paint,
4. adds lo-fi degradation,
5. returns a PNG.

## Requirements

- Python **3.11**
- macOS/Linux/Windows

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -F "image=@/path/to/photo.jpg" \
  -F "text=I miss you" \
  -F "paint_thickness=18" \
  -F "messiness=0.65" \
  -F "text_wobble=0.6" \
  --output out.png
```

## Notes

- Deterministic outputs: pass `seed` to `/generate`.
- No diffusion models.
- Text is rendered exactly as provided.
- Segmentation uses a pretrained torchvision model by default (DeepLabV3). The module is structured so you can swap in U²-Net later.
