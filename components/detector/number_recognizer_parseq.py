"""
Jersey number recognition using PARSeq STR model (jersey-number-pipeline/str.py style).
For each bbox the region is cropped, preprocessed and passed to the parseq checkpoint.
"""

from __future__ import annotations

import os
import re
import string
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from common.classes.number import Number

# basketballs repo root (components/detector -> components -> basketballs)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _REPO_ROOT / "models"
# PARSeq checkpoint in models/ (e.g. parseq_epoch=24-....ckpt); no parseq.pt there
_DEFAULT_CHECKPOINT = next(_MODELS_DIR.glob("parseq*.ckpt"), _MODELS_DIR / "parseq.pt")
# Path to parseq package for strhub imports (use without pip install)
_PARSEQ_ROOT = _REPO_ROOT / "components" / "str" / "parseq"

_model = None
_transform = None
_device = "cuda"
_save_crop_counter = 0


def _ensure_parseq_path():
    """Add parseq package to sys.path so strhub can be imported without installing."""
    if str(_PARSEQ_ROOT) not in sys.path and _PARSEQ_ROOT.exists():
        sys.path.insert(0, str(_PARSEQ_ROOT))


def _get_model(checkpoint_path: str | Path | None = None):
    global _model, _transform, _device
    if _model is not None:
        return _model
    _ensure_parseq_path()
    try:
        from strhub.data.module import SceneTextDataModule
        from strhub.models.utils import load_from_checkpoint
    except ImportError as e:
        raise ImportError(
            f"PARSeq STR not found. Add jersey-number-pipeline/str/parseq to PYTHONPATH or set PARSEQ_ROOT. {e}"
        ) from e
    path = checkpoint_path or _DEFAULT_CHECKPOINT
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # PyTorch 2.6+ defaults to weights_only=True; Lightning checkpoints need weights_only=False
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    print(f"[number_recognizer_parseq] Loading PARSeq checkpoint: {path}")
    kwargs = {"charset_test": string.digits}
    global _device
    _model = load_from_checkpoint(str(path), **kwargs).eval()
    try:
        _model = _model.to(_device)
        print(f"[number_recognizer_parseq] Model loaded on device: {_device}")
    except Exception:
        _device = "cpu"
        _model = _model.to("cpu")
        print(f"[number_recognizer_parseq] Model loaded on device: {_device} (cuda unavailable)")
    _transform = SceneTextDataModule.get_transform(_model.hparams.img_size)
    print(f"[number_recognizer_parseq] Ready. img_size={getattr(_model.hparams, 'img_size', '?')}")
    return _model


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Light preprocessing: keep color (BGR), optional light denoise."""
    if crop.size == 0:
        return crop
    img = crop.copy() if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
    return img


# Number of augmented runs; recognition accepted only when all runs agree
_N_VOTES = 5


def _augment_crop(crop: np.ndarray, variant: int) -> np.ndarray:
    """Apply a deterministic augmentation variant (0..4) for voting."""
    if crop.size == 0:
        return crop
    out = crop.copy()
    h, w = out.shape[:2]
    if variant == 0:
        return out
    if variant == 1:
        # Slight rotation one way
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=4.0, scale=1.0)
        return cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if variant == 2:
        # Slight brightness/contrast
        out = cv2.convertScaleAbs(out, alpha=1.1, beta=8)
        return out
    if variant == 3:
        # Slight rotation the other way
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=-4.0, scale=1.0)
        return cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if variant == 4:
        # Light Gaussian noise
        noise = np.random.RandomState(42 + variant).randn(*out.shape).astype(np.float32) * 8
        out = cv2.add(out.astype(np.float32), noise)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out
    return out


def _parse_digit(text: str) -> int | None:
    """Extract a single digit 0–9 or two-digit number (0–99) from model output."""
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


def _predict_crop(model, transform, crop_bgr: np.ndarray, device: str):
    """Run PARSeq on one crop (BGR numpy). Returns (predicted_str, confidence float)."""
    import torch
    from PIL import Image
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs_full = logits[:, :3, :11].softmax(-1)
    preds, probs = model.tokenizer.decode(probs_full)
    pred_str = preds[0]
    conf = probs[0].cpu().detach().numpy().squeeze()
    if hasattr(conf, "tolist"):
        conf = conf.tolist()
    if isinstance(conf, (list, np.ndarray)):
        conf_value = float(np.prod(conf)) if len(conf) else 0.0
    else:
        conf_value = float(conf)
    return pred_str, conf_value


def recognize_numbers_in_frame(
    frame: np.ndarray,
    number_detections: List[Number],
    *,
    padding: int = 5,
    ocr_conf_threshold: float = 0.999,
    save_crops_dir: str | Path | None = None,
    frame_id: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> List[Number]:
    """
    Recognize digit/number in each bbox using PARSeq model. Sets `num` when confidence >= ocr_conf_threshold.

    Args:
        frame: BGR frame (OpenCV).
        number_detections: list of detections with bbox [x1, y1, x2, y2].
        padding: pixels to expand bbox when cropping (typically 2–5).
        ocr_conf_threshold: minimum confidence (0.0–1.0). Below this, num is not set.
        save_crops_dir: if set, save each preprocessed crop to this directory.
        frame_id: optional frame index for saved filenames.
        checkpoint_path: path to parseq checkpoint (e.g. models/parseq.pt). Default: REPO_ROOT/models/parseq.pt.

    Returns:
        The same list `number_detections` with `num` filled (int 0–99 or None).
    """
    global _model, _transform, _save_crop_counter
    model = _get_model(checkpoint_path)

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
            votes: list[int | None] = []
            confs: list[float] = []
            for v in range(_N_VOTES):
                aug = _augment_crop(preprocessed, v)
                pred_str, conf = _predict_crop(_model, _transform, aug, _device)
                parsed = _parse_digit(pred_str)
                if parsed is not None and conf >= ocr_conf_threshold:
                    votes.append(parsed)
                    confs.append(conf)
                else:
                    break
                if votes[-1] != votes[0]:
                    break
                
            if len(votes) != _N_VOTES or not all(x is not None for x in votes) or len(set(votes)) != 1:
                number.num = None
            else:
                number.num = votes[0]
                number.confidence = sum(confs) / len(confs)
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
