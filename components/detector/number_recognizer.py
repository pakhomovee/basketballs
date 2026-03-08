"""
Jersey number recognition from frame and list of number detections (bbox).
For each bbox the region is cropped, preprocessed and passed to OCR (EasyOCR).
"""

from __future__ import annotations

import re
from typing import List

import cv2
import numpy as np

from common.classes.number import Number

try:
    import easyocr
except ImportError:
    easyocr = None  # type: ignore

# Lazy init of Reader (heavy model load on first use)
_reader: "easyocr.Reader | None" = None


def _get_reader() -> "easyocr.Reader | None":
    global _reader
    if easyocr is None:
        return None
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Grayscale, contrast, threshold — for better digit recognition."""
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape[:2]
    min_side = 20
    if w < min_side or h < min_side:
        scale = max(min_side / w, min_side / h, 1.5)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if gray.size > 0:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray


def _parse_digit(text: str) -> int | None:
    """Extract a single digit 0–9 or two-digit number (0–99) from OCR string."""
    if not text:
        return None
    digits = re.sub(r"\D", "", text.strip())
    if not digits:
        return None
    num = int(digits[:2]) if len(digits) >= 2 else int(digits[0])
    if 0 <= num <= 99:
        return num
    if 0 <= int(digits[0]) <= 9:
        return int(digits[0])
    return None


def recognize_numbers_in_frame(
    frame: np.ndarray,
    number_detections: List[Number],
    *,
    padding: int = 30,
    ocr_conf_threshold: float = 0.99,
) -> List[Number]:
    """
    Recognize digit/number in each bbox region from frame and list of number detections.
    Sets `num` only when OCR confidence is not below ocr_conf_threshold.

    Args:
        frame: BGR frame (OpenCV).
        number_detections: list of detections with bbox [x1, y1, x2, y2].
        padding: pixels to expand bbox when cropping.
        ocr_conf_threshold: minimum EasyOCR confidence (0.0–1.0). Below this, num is not set.

    Returns:
        The same list `number_detections` with `num` filled (int 0–99 or None).
    """
    reader = _get_reader()
    if reader is None:
        for n in number_detections:
            n.num = None
        return number_detections

    h, w = frame.shape[:2]
    for number in number_detections:
        x1, y1, x2, y2 = number.bbox[0], number.bbox[1], number.bbox[2], number.bbox[3]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            number.num = None
            continue
        crop = frame[y1:y2, x1:x2]
        preprocessed = _preprocess_crop(crop)
        try:
            results = reader.readtext(preprocessed, allowlist="0123456789")
            best_num: int | None = None
            best_conf: float = -1.0
            for _bbox, text, conf in results:
                parsed = _parse_digit(text)
                if parsed is not None and conf > best_conf:
                    best_num = parsed
                    best_conf = conf
            number.num = best_num if best_conf >= ocr_conf_threshold else None
        except Exception:
            number.num = None
    return number_detections
