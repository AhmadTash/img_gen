from __future__ import annotations

import argparse
from pathlib import Path

from app.pipeline.run import generate_image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    raw = Path(args.in_path).read_bytes()
    out = generate_image(raw_image_bytes=raw, text=args.text, seed=args.seed)
    Path(args.out_path).write_bytes(out)


if __name__ == "__main__":
    main()
