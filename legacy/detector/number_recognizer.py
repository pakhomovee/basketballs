"""
Jersey number recognition from frame and list of number detections (bbox).
For each bbox the region is cropped, preprocessed and passed to OCR (EasyOCR).
"""

from __future__ import annotations

import re
from pathlib import Path
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
# Counter for saved crops when frame_id is not provided
_save_crop_counter = 0


def _get_reader() -> "easyocr.Reader | None":
    global _reader
    if easyocr is None:
        return None
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """
    Light preprocessing: keep color (BGR), optional light denoise.
    """
    if crop.size == 0:
        return crop
    img = crop.copy() if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
    return img


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
    padding: int = 5,
    ocr_conf_threshold: float = 0.99,
    save_crops_dir: str | Path | None = None,
    frame_id: int | None = None,
) -> List[Number]:
    """
    Recognize digit/number in each bbox region from frame and list of number detections.
    Sets `num` only when OCR confidence is not below ocr_conf_threshold.

    Args:
        frame: BGR frame (OpenCV).
        number_detections: list of detections with bbox [x1, y1, x2, y2].
        padding: pixels to expand bbox when cropping (typically 2–5).
        ocr_conf_threshold: minimum EasyOCR confidence (0.0–1.0). Below this, num is not set.
        save_crops_dir: if set, save each preprocessed crop sent to the model to this directory.
        frame_id: optional frame index, used in saved filenames when save_crops_dir is set.

    Returns:
        The same list `number_detections` with `num` filled (int 0–99 or None).
    """
    global _save_crop_counter
    reader = _get_reader()
    if reader is None:
        for n in number_detections:
            n.num = None
        return number_detections

    if save_crops_dir is not None:
        save_path = Path(save_crops_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    h, w = frame.shape[:2]
    for i, number in enumerate(number_detections):
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
        if save_crops_dir is not None:
            if frame_id is not None:
                base = f"frame_{frame_id:06d}_crop_{i:02d}"
            else:
                base = f"call_{_save_crop_counter:06d}_crop_{i:02d}"
            num_suffix = f"_num_{number.num}" if number.num is not None else "_num_None"
            cv2.imwrite(str(save_path / f"{base}{num_suffix}.png"), preprocessed)
            cv2.imwrite(str(save_path / f"{base}{num_suffix}_raw.png"), crop)
    if save_crops_dir is not None and frame_id is None:
        _save_crop_counter += 1
    return number_detections
