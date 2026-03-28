"""Training loop for ReID with warmup + cosine annealing LR schedule."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PKSampler, SynergyReIDDataset, build_train_transform
from .evaluate import evaluate
from .losses import ArcFaceLoss, TripletLoss
from .model import ReIDModel

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _lr_scale(epoch: int, *, warmup_epochs: int, total_epochs: int, eta_min: float, base_lr: float) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_scale = eta_min / base_lr
    return min_scale + (1 - min_scale) * cosine


@dataclass
class EpochStats:
    id_loss: float = 0.0
    triplet_loss: float = 0.0
    total_loss: float = 0.0

    def update(self, *, id_loss: float, triplet_loss: float, total_loss: float) -> None:
        self.id_loss += id_loss
        self.triplet_loss += triplet_loss
        self.total_loss += total_loss

    def averaged(self, steps: int) -> tuple[float, float, float]:
        denom = max(steps, 1)
        return self.id_loss / denom, self.triplet_loss / denom, self.total_loss / denom


def _create_train_loader(train_dir: Path, *, p: int, k: int, num_workers: int) -> tuple[SynergyReIDDataset, DataLoader]:
    train_ds = SynergyReIDDataset(train_dir, transform=build_train_transform())
    sampler = PKSampler(train_ds, p=p, k=k)
    train_loader = DataLoader(
        train_ds,
        batch_size=p * k,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_ds, train_loader


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-7,
) -> torch.optim.lr_scheduler.LambdaLR:
    lambdas = []
    for group in optimizer.param_groups:
        base_lr = group["lr"]
        lambdas.append(
            lambda epoch, _bl=base_lr: _lr_scale(
                epoch,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                eta_min=eta_min,
                base_lr=_bl,
            )
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def _run_epoch(
    model: ReIDModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    id_loss: ArcFaceLoss,
    tri_loss: TripletLoss,
    *,
    epoch: int,
    device: str,
) -> tuple[EpochStats, float]:
    model.train()
    stats = EpochStats()
    start_time = time.time()

    for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        global_feat, normed_bn_feat = model(imgs)
        loss_id = id_loss(normed_bn_feat, labels)
        loss_tri = tri_loss(global_feat, labels)
        loss = loss_id + loss_tri

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        stats.update(
            id_loss=loss_id.item(),
            triplet_loss=loss_tri.item(),
            total_loss=loss.item(),
        )

    return stats, time.time() - start_time


def _log_epoch(
    epoch: int, epochs: int, optimizer: torch.optim.Optimizer, stats: EpochStats, steps: int, elapsed: float
) -> None:
    avg_id, avg_triplet, avg_total = stats.averaged(steps)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info(
        "Epoch %3d/%d  lr=%.2e  id=%.4f  tri=%.4f  total=%.4f  (%.1fs)",
        epoch + 1,
        epochs,
        cur_lr,
        avg_id,
        avg_triplet,
        avg_total,
        elapsed,
    )


def _evaluate_and_maybe_save(
    model: ReIDModel,
    *,
    test_query: Path,
    test_gallery: Path,
    output_dir: Path,
    device: str,
    best_map: float,
) -> float:
    metrics = evaluate(model, str(test_query), str(test_gallery), device=device)
    logger.info(
        "  => mAP=%.2f%%  rank-1=%.2f%%  rank-5=%.2f%%  rank-10=%.2f%%",
        metrics["mAP"] * 100,
        metrics["rank-1"] * 100,
        metrics["rank-5"] * 100,
        metrics["rank-10"] * 100,
    )
    if metrics["mAP"] <= best_map:
        return best_map

    ckpt_path = output_dir / "best_model.pth"
    torch.save(model.state_dict(), ckpt_path)
    logger.info("  => New best mAP! Saved to %s", ckpt_path)
    return metrics["mAP"]


def _has_eval_split(query_dir: Path, gallery_dir: Path) -> bool:
    return query_dir.exists() and gallery_dir.exists()


def train(
    data_root: str,
    output_dir: str = "logs/reid",
    epochs: int = 120,
    p: int = 16,
    k: int = 4,
    lr: float = 3.5e-4,
    device: str = "cuda",
) -> ReIDModel:
    """Train a ReID model on SynergyReID data."""
    data_root = Path(data_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "reid_training"
    test_query = data_root / "reid_test" / "query"
    test_gallery = data_root / "reid_test" / "gallery"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}. "
            "Expected --data_root to contain reid_training/ and optionally reid_test/query and reid_test/gallery/."
        )

    num_workers = 4
    warmup_epochs = 10
    eval_interval = 10

    logger.info("Loading training data from %s", train_dir)
    train_ds, train_loader = _create_train_loader(train_dir, p=p, k=k, num_workers=num_workers)
    logger.info("Training set: %d images, %d identities", len(train_ds), train_ds.num_pids)

    model = ReIDModel(pretrained=True).to(device)
    id_loss = ArcFaceLoss(num_classes=train_ds.num_pids).to(device)
    tri_loss = TripletLoss().to(device)

    # ArcFace head starts from random — give it a 10× higher LR than the backbone.
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": lr},
            {"params": id_loss.parameters(), "lr": lr * 10},
        ],
        weight_decay=5e-4,
    )
    scheduler = _create_scheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs)
    has_eval_split = _has_eval_split(test_query, test_gallery)
    if not has_eval_split:
        logger.info("Evaluation skipped: %s or %s does not exist", test_query, test_gallery)

    best_mAP = 0.0
    iters_per_epoch = len(train_loader)

    scheduler.step()  # apply warmup scaling before first epoch
    for epoch in range(epochs):
        stats, elapsed = _run_epoch(
            model,
            train_loader,
            optimizer,
            id_loss,
            tri_loss,
            epoch=epoch,
            device=device,
        )
        _log_epoch(epoch, epochs, optimizer, stats, iters_per_epoch, elapsed)
        scheduler.step()

        if has_eval_split and ((epoch + 1) % eval_interval == 0 or epoch == epochs - 1):
            best_mAP = _evaluate_and_maybe_save(
                model,
                test_query=test_query,
                test_gallery=test_gallery,
                output_dir=out,
                device=device,
                best_map=best_mAP,
            )

    final_path = out / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    logger.info("Training done. Final model saved to %s", final_path)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ReID model (ArcFace + triplet)")
    parser.add_argument("data_root", help="Path to dataset root (with reid_training/ and reid_test/)")
    parser.add_argument("-o", "--output_dir", default="logs/reid")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--p", type=int, default=16, help="Identities per batch")
    parser.add_argument("--k", type=int, default=4, help="Images per identity per batch")
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger.info("Starting ReID training: data_root=%s device=%s epochs=%d", args.data_root, args.device, args.epochs)

    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        p=args.p,
        k=args.k,
        lr=args.lr,
        device=args.device,
    )
