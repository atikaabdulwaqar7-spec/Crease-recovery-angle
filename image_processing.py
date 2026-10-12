"""
image_processing.py
-------------------
Loading, resizing and segmenting the fabric specimen from a photo.
"""

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(file) -> np.ndarray:
    """Read an uploaded file (Streamlit UploadedFile / path) into a BGR numpy array.
    Phone-camera EXIF rotation is applied automatically."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_max(img: np.ndarray, max_dim: int = 1000):
    """Shrink the image so its longest side is <= max_dim.
    Returns (resized_image, scale) where scale = new_size / old_size."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def _border_fraction(binary: np.ndarray) -> float:
    """Fraction of the image border that is white (foreground)."""
    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    return float(np.count_nonzero(border)) / border.size


def segment_fabric(
    img: np.ndarray,
    blur: int = 5,
    method: str = "otsu",
    threshold: int = 127,
    polarity: str = "auto",
    morph: int = 5,
    min_area_frac: float = 0.002,
) -> np.ndarray:
    """
    Separate the fabric from the background and return a binary mask
    (255 = fabric, 0 = background).

    method   : "otsu" | "adaptive" | "manual"
    polarity : "auto"           -> picks the side that touches the image border less
               "fabric_bright"  -> fabric lighter than background
               "fabric_dark"    -> fabric darker than background
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = max(1, blur | 1)  # kernel must be odd
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5
        )
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # binary currently = "bright" pixels are white
    if polarity == "fabric_dark":
        binary = cv2.bitwise_not(binary)
    elif polarity == "auto":
        if _border_fraction(binary) > _border_fraction(cv2.bitwise_not(binary)):
            binary = cv2.bitwise_not(binary)

    # Clean up noise
    if morph > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # Keep only reasonably large blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = min_area_frac * binary.size
    clean = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean
