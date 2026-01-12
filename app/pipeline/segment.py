from __future__ import annotations

from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision


@lru_cache(maxsize=1)
def _load_segmentation_model() -> torch.nn.Module:
    """Load a pretrained person segmentation model.

    We use torchvision's DeepLabV3-ResNet50 pretrained on COCO/VOC-style classes.
    It's not U²-Net, but it satisfies the requirement of using a pretrained person
    segmentation model without training.

    Swap-in point: replace this with U²-Net weights or another model.
    """

    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    model.eval()
    return model


def segment_person_mask(rgb_u8: np.ndarray, *, seed: Optional[int] = None) -> np.ndarray:
    """Return a binary person mask (uint8 0/255) of shape HxW.

    Steps:
    - normalize to the model's expected input
    - run segmentation
    - extract "person" class
    - threshold and clean (opening + remove tiny specks)
    """

    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("segment_person_mask expects RGB uint8 HxWx3")

    # Determinism: inference is deterministic by default on CPU; we also set seeds
    # so any future randomness (or GPU kernels) are consistent.
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))

    device = torch.device("cpu")
    model = _load_segmentation_model().to(device)
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    pil = torchvision.transforms.functional.to_pil_image(rgb_u8)
    x = preprocess(pil).unsqueeze(0).to(device)  # 1x3xHxW

    with torch.inference_mode():
        out = model(x)["out"]  # 1xCxhxw
        # DeepLabV3 is a *multi-class* semantic segmentation head.
        # The correct way to get a person mask is argmax over classes then pick the
        # person label index (15 for the VOC-style mapping used by these weights).
        pred = out.argmax(dim=1)[0]  # [h,w] class indices

    pred_np = pred.cpu().numpy().astype(np.int32)

    # Resize back to input size (transforms may rescale)
    pred_np = cv2.resize(
        pred_np,
        (rgb_u8.shape[1], rgb_u8.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    person_idx = 15
    mask = (pred_np == person_idx).astype(np.uint8) * 255

    # Cleanup: open removes isolated dots; then keep only sufficiently large areas.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # Remove tiny connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    min_area = max(200, int(0.0005 * mask.shape[0] * mask.shape[1]))
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned
