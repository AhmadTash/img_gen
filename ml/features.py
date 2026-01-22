import cv2
import numpy as np

def extract_features(image_bytes: bytes) -> dict:
    """
    Extracts visual features from an image byte stream.
    Features:
      - mean_brightness (0-255)
      - contrast (std dev of V channel)
      - mean_saturation (0-255)
      - warmth (Red / Blue ratio)
      - sharpness (Laplacian variance)
    """
    # clear checks
    if not image_bytes:
        raise ValueError("Empty image bytes")

    # Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not decode image")

    # Convert to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # 1. mean_brightness
    mean_brightness = np.mean(v)

    # 2. contrast (std dev of V)
    contrast = np.std(v)

    # 3. mean_saturation
    mean_saturation = np.mean(s)

    # 4. warmth (R/B ratio)
    # Avoid div by zero
    b_channel, _, r_channel = cv2.split(img_bgr)
    mean_r = np.mean(r_channel)
    mean_b = np.mean(b_channel) + 1e-6
    warmth = mean_r / mean_b

    # 5. sharpness (Laplacian var)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "mean_brightness": float(mean_brightness),
        "contrast": float(contrast),
        "mean_saturation": float(mean_saturation),
        "warmth": float(warmth),
        "sharpness": float(sharpness),
    }
