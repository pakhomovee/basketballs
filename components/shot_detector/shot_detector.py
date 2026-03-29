"""
Shot detector: MS-TCN-based frame-level classification of
*nothing / shot / make* using pre-computed clip features.

Provides:
  - :class:`ShotDetector`  – loads a single ``.pt`` checkpoint (``meta`` + ``state_dict``),
    uses ``AppConfig`` (``cfg.main.court_type``); primary API is
    :meth:`~ShotDetector.predict_from_detections`.
  - :func:`ms_tcn_loss`    – combined CE + truncated-MSE smoothing loss
  - :func:`train`          – full training pipeline (CLI-runnable)
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from common.classes import CourtType
from common.classes.ball import Ball
from common.classes.detections import Detection
from common.utils.models import get_model_paths
from common.utils.utils import get_device
from config import AppConfig, load_default_config
from shot_detector.dataset import (
    NUM_CLASSES,
    ShotDataset,
    StackedShotDataset,
    collate_fn,
)
from shot_detector.model import MultiStageTCN
from shot_detector.shot_embedder import ShotEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SHOT_CHECKPOINT_FILENAME = "shot_detection_model.pt"
SHOT_META_KEYS = ("input_dim", "n_classes", "n_stages", "n_filters", "n_layers")


def _shot_meta_dict(
    input_dim: int,
    n_classes: int,
    n_stages: int,
    n_filters: int,
    n_layers: int,
) -> dict[str, int]:
    return {
        "input_dim": int(input_dim),
        "n_classes": int(n_classes),
        "n_stages": int(n_stages),
        "n_filters": int(n_filters),
        "n_layers": int(n_layers),
    }


def save_shot_checkpoint(path: str | Path, meta: dict[str, int], state_dict: dict) -> None:
    """Write ``{ "meta": {...}, "state_dict": ... }`` (``meta`` must match :data:`SHOT_META_KEYS`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for k in SHOT_META_KEYS:
        if k not in meta:
            raise KeyError(f"meta missing {k!r}")
    torch.save({"meta": {k: int(meta[k]) for k in SHOT_META_KEYS}, "state_dict": state_dict}, path)


def load_shot_checkpoint(path: str | Path) -> tuple[dict[str, int], dict]:
    path = Path(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "meta" not in raw or "state_dict" not in raw:
        raise ValueError(f"Expected a dict with 'meta' and 'state_dict' keys, got {type(raw).__name__}")
    meta_raw = raw["meta"]
    if not isinstance(meta_raw, dict):
        raise TypeError("checkpoint['meta'] must be a dict")
    meta = {k: int(meta_raw[k]) for k in SHOT_META_KEYS}
    return meta, raw["state_dict"]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ms_tcn_loss(
    stage_outputs: list[torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    lambda_smooth: float = 0.15,
    tau: float = 4.0,
) -> torch.Tensor:
    """
    MS-TCN loss summed over all stages.

    Per-stage loss = cross-entropy + λ · truncated-MSE on log-probabilities.

    Parameters
    ----------
    stage_outputs : list of (B, C, T) logits (one per stage).
    labels        : (B, T) ground-truth class indices.
    mask          : (B, T) bool – True for valid positions.
    class_weights : optional (C,) tensor for weighted CE.
    lambda_smooth : smoothing-loss weight (paper default 0.15).
    tau           : truncation threshold  (paper default 4).
    """
    total = torch.tensor(0.0, device=labels.device)
    mask_f = mask.float()
    valid_count = mask_f.sum().clamp(min=1.0)

    for out in stage_outputs:
        # --- classification loss (masked) ---
        ce = F.cross_entropy(out, labels, weight=class_weights, reduction="none")  # (B, T)
        ce = (ce * mask_f).sum() / valid_count

        # --- smoothing loss: truncated MSE on log-probs ---
        log_probs = F.log_softmax(out, dim=1)  # (B, C, T)
        # Δ_{t,c} = |log ŷ_{t,c} − log ŷ_{t-1,c}|  (detach t-1)
        delta = torch.abs(log_probs[:, :, 1:] - log_probs[:, :, :-1].detach())
        delta = torch.clamp(delta, max=tau)

        smooth_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(1).float()  # (B, 1, T-1)
        C = out.shape[1]
        t_mse = (delta**2 * smooth_mask).sum() / (smooth_mask.sum() * C).clamp(min=1.0)

        total = total + ce + lambda_smooth * t_mse

    return total


# ---------------------------------------------------------------------------
# ShotDetector (inference)
# ---------------------------------------------------------------------------

class ShotDetector:
    """MS-TCN shot / make segmentation from ball, rim, and homography inputs.

    Like :class:`court_detector.court_detector.CourtDetector`, this class takes
    an :class:`~config.AppConfig` (defaulting to the main YAML) and reads
    ``cfg.main.court_type`` for :class:`~shot_detector.shot_embedder.ShotEmbedder`.

    Weights are loaded from *model_path* (single ``.pt``: ``meta`` + ``state_dict``).
    If *model_path* is ``None``, uses ``cfg`` models section (``shot_detection_model.pt``).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        cfg: AppConfig | None = None,
        device: str | None = None,
    ):
        if cfg is None:
            cfg = load_default_config()
        self.cfg = cfg
        self.device = device or get_device()

        ct = cfg.main.court_type.lower()
        self.court_type = CourtType.NBA if ct == "nba" else CourtType.FIBA
        self.embedder = ShotEmbedder(court_type=self.court_type)

        path = Path(model_path) if model_path is not None else get_model_paths(cfg).shot_detection
        meta, state = load_shot_checkpoint(path)
        self.model = MultiStageTCN(
            input_dim=meta["input_dim"],
            n_classes=meta["n_classes"],
            n_stages=meta["n_stages"],
            n_filters=meta["n_filters"],
            n_layers=meta["n_layers"],
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _forward_embedding(self, embedding: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(embedding.astype(np.float32, copy=False)).unsqueeze(0)
        x = x.permute(0, 2, 1).to(self.device)
        outputs = self.model(x)
        logits = outputs[-1]
        return logits.argmax(dim=1).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_from_detections(
        self,
        ball_detections: dict[int, Ball | list[Ball]],
        rim_detections: dict[int, list[Detection]],
        homographies: list[np.ndarray | None] | dict[int, np.ndarray | None],
        *,
        frame_width: float,
        frame_height: float,
        num_frames: int | None = None,
    ) -> np.ndarray:
        """
        Predict per-frame labels from raw detections + homographies (preferred entry point).

        *frame_width* / *frame_height* must match the video (same capture as homographies).

        Returns
        -------
        np.ndarray
            Shape ``(T,)``, values in ``{0, 1, 2}`` — nothing / shot / make.
        """
        embedding = self.embedder.build_embedding(
            ball_detections,
            rim_detections,
            homographies,
            frame_width=frame_width,
            frame_height=frame_height,
            num_frames=num_frames,
        )
        return self._forward_embedding(embedding)

    @torch.no_grad()
    def predict_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Predict from an already-built ``(T, 112)`` embedding (e.g. tests / pipelines)."""
        return self._forward_embedding(embedding)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_class_weights(dataset: ShotDataset) -> torch.Tensor:
    """Inverse-frequency class weights computed over the full dataset."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in range(len(dataset)):
        _, labels, T = dataset[i]
        for c in range(NUM_CLASSES):
            counts[c] += (labels[:T] == c).sum().item()

    total = counts.sum()
    weights = np.where(counts > 0, total / (NUM_CLASSES * counts), 1.0)
    return torch.from_numpy(weights).float()


@torch.no_grad()
def _evaluate(
    model: MultiStageTCN,
    loader: DataLoader,
    device: str,
    class_weights: torch.Tensor | None,
    lambda_smooth: float,
    tau: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)

    for features, labels, mask in loader:
        features = features.permute(0, 2, 1).to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        outputs = model(features, mask)
        loss = ms_tcn_loss(
            outputs, labels, mask,
            class_weights=class_weights, lambda_smooth=lambda_smooth, tau=tau,
        )
        total_loss += loss.item()

        preds = outputs[-1].argmax(dim=1)  # (B, T)
        valid = mask.bool()
        correct += ((preds == labels) & valid).sum().item()
        total += valid.sum().item()

        for c in range(NUM_CLASSES):
            c_mask = (labels == c) & valid
            per_class_correct[c] += ((preds == c) & c_mask).sum().item()
            per_class_total[c] += c_mask.sum().item()

    n_batches = max(len(loader), 1)
    per_class_acc = np.where(
        per_class_total > 0, per_class_correct / per_class_total, 0.0,
    )
    return {
        "loss": total_loss / n_batches,
        "accuracy": correct / max(total, 1),
        "per_class_acc": per_class_acc,
    }


def train(
    features_dir: str | Path,
    output_dir: str | Path,
    *,
    n_stages: int = 4,
    n_filters: int = 64,
    n_layers: int = 10,
    dropout: float = 0.5,
    lr: float = 5e-4,
    num_epochs: int = 50,
    batch_size: int = 1,
    num_workers: int = 10,
    lambda_smooth: float = 0.15,
    tau: float = 4.0,
    val_fraction: float = 0.2,
    seed: int = 42,
    court_type: str = "nba",
    fliplr: bool = True,
    random_scale: float = 1.1,
    random_shift: float = 0.02,
    random_rotate: float = 3.0,
    skip_prob: float = 0.2,
    random_crop_ratio: float = 1.0,
    max_stack: int = 3,
    cfg: AppConfig | None = None,
    device: str | None = None,
) -> ShotDetector:
    """
    Full MS-TCN training pipeline.

    When *max_stack* ≥ 2, each training sample concatenates ``k`` clips in time
    with ``k`` uniform in ``{1, …, max_stack}`` (see :class:`~shot_detector.dataset.StackedShotDataset`).
    *random_crop_ratio* < 1 applies a random temporal crop per clip **before** stacking
    (see :class:`~shot_detector.dataset.ShotDataset`). Validation uses full clips (no crop / skip).

    Returns a :class:`ShotDetector` wrapping the best model.
    """
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or get_device()

    if cfg is None:
        cfg = load_default_config()
    ct_norm = court_type.lower()
    cfg = cfg.model_copy(update={"main": cfg.main.model_copy(update={"court_type": ct_norm})})
    ct = CourtType.NBA if ct_norm == "nba" else CourtType.FIBA

    # ---- datasets ----
    train_ds = ShotDataset(
        features_dir, court_type=ct,
        fliplr=fliplr, random_scale=random_scale,
        random_shift=random_shift, random_rotate=random_rotate,
        skip_prob=skip_prob,
        random_crop_ratio=random_crop_ratio,
    )
    val_ds = ShotDataset(features_dir, court_type=ct)

    n_total = len(train_ds)
    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n_total * val_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    log.info("Dataset: %d total, %d train, %d val", n_total, len(train_indices), len(val_indices))

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)

    if max_stack < 1:
        raise ValueError("max_stack must be >= 1")
    train_loader_ds: Dataset = (
        StackedShotDataset(train_subset, max_stack) if max_stack >= 2 else train_subset
    )
    log.info(
        "Train: %s (max_stack=%d); val: single clips",
        f"1..{max_stack} clips concatenated" if max_stack >= 2 else "one clip per sample",
        max_stack,
    )

    train_loader = DataLoader(train_loader_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)

    # ---- class weights (full timeline; skip_prob would bias frequencies) ----
    log.info("Computing class weights …")
    weights_count_ds = ShotDataset(
        features_dir,
        court_type=ct,
        fliplr=False,
        random_scale=1.0,
        random_shift=0.0,
        random_rotate=0.0,
        skip_prob=0.0,
        random_crop_ratio=1.0,
    )
    class_weights = _compute_class_weights(Subset(weights_count_ds, train_indices)).to(device)
    log.info("Class weights: %s", class_weights.cpu().numpy())

    # ---- model ----
    input_dim = ShotEmbedder.FLAT_DIM
    model = MultiStageTCN(
        input_dim=input_dim,
        n_classes=NUM_CLASSES,
        n_stages=n_stages,
        n_filters=n_filters,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %d params, device=%s", n_params, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ---- training loop ----
    best_val_loss = math.inf
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for features, labels, mask in train_loader:
            features = features.permute(0, 2, 1).to(device)  # (B, D, T)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(features, mask)
            loss = ms_tcn_loss(
                outputs, labels, mask,
                class_weights=class_weights, lambda_smooth=lambda_smooth, tau=tau,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()

        val_metrics = _evaluate(model, val_loader, device, class_weights, lambda_smooth, tau)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        pca = val_metrics["per_class_acc"]

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        cur_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d  lr=%.2e  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  "
            "bg=%.3f shot=%.3f make=%.3f%s",
            epoch, num_epochs, cur_lr, train_loss, val_loss, val_acc,
            pca[0], pca[1], pca[2],
            " *" if improved else "",
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "per_class_acc": pca.tolist(),
        })

    checkpoint_path = output_dir / SHOT_CHECKPOINT_FILENAME
    state_to_save = best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()}
    meta_pt = _shot_meta_dict(input_dim, NUM_CLASSES, n_stages, n_filters, n_layers)
    save_shot_checkpoint(checkpoint_path, meta_pt, state_to_save)

    with (output_dir / "history.jsonl").open("w", encoding="utf-8") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")

    log.info(
        "Training done. Best val loss: %.4f. Checkpoint: %s",
        best_val_loss,
        checkpoint_path,
    )

    return ShotDetector(checkpoint_path, cfg=cfg, device=device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    default_features = Path(__file__).resolve().parent / "dataset_features"
    default_output = Path(__file__).resolve().parent / "trained_model"

    p = argparse.ArgumentParser(description="Train MS-TCN shot detector")
    p.add_argument("--features-dir", type=Path, default=default_features)
    p.add_argument("--output-dir", type=Path, default=default_output)

    g = p.add_argument_group("model")
    g.add_argument("--n-stages", type=int, default=4)
    g.add_argument("--n-filters", type=int, default=64)
    g.add_argument("--n-layers", type=int, default=10)
    g.add_argument("--dropout", type=float, default=0.5)

    g = p.add_argument_group("training")
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--num-workers", type=int, default=10)
    g.add_argument("--lambda-smooth", type=float, default=0.15)
    g.add_argument("--tau", type=float, default=4.0)
    g.add_argument("--val-fraction", type=float, default=0.2)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("data")
    g.add_argument("--court-type", type=str, default="nba", choices=["nba", "fiba"])

    g = p.add_argument_group("augmentation")
    g.add_argument("--fliplr", action="store_true", default=True)
    g.add_argument("--no-fliplr", dest="fliplr", action="store_false")
    g.add_argument("--random-scale", type=float, default=1.5)
    g.add_argument("--random-shift", type=float, default=0.5)
    g.add_argument("--random-rotate", type=float, default=20.0)
    g.add_argument("--skip-prob", type=float, default=0.0, help="Train-only random frame drop probability")
    g.add_argument(
        "--random-crop-ratio",
        type=float,
        default=1.0,
        help="Train only: temporal window length u*T with u~U[ratio,1]; 1.0 disables",
    )
    g.add_argument(
        "--max-stack",
        type=int,
        default=3,
        help="Train only: concatenate 1..max_stack random clips per sample (1 disables stacking)",
    )

    g = p.add_argument_group("device")
    g.add_argument("--device", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        n_stages=args.n_stages,
        n_filters=args.n_filters,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lambda_smooth=args.lambda_smooth,
        tau=args.tau,
        val_fraction=args.val_fraction,
        seed=args.seed,
        court_type=args.court_type,
        fliplr=args.fliplr,
        random_scale=args.random_scale,
        random_shift=args.random_shift,
        random_rotate=args.random_rotate,
        skip_prob=args.skip_prob,
        random_crop_ratio=args.random_crop_ratio,
        max_stack=args.max_stack,
        device=args.device,
    )
