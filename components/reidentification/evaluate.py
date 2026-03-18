"""Evaluation utilities for person re-identification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import QueryGalleryDataset
from .model import ReIDModel


def _create_eval_loader(root: str | Path, *, batch_size: int, num_workers: int) -> DataLoader:
    dataset = QueryGalleryDataset(root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.inference_mode()
def _extract_embeddings(
    model: ReIDModel, loader: DataLoader, *, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    pids: list[torch.Tensor] = []
    camids: list[torch.Tensor] = []

    for images, batch_pids, batch_camids in loader:
        images = images.to(device, non_blocking=True)
        features.append(model.extract_features(images).cpu())
        pids.append(torch.as_tensor(batch_pids, dtype=torch.int64))
        camids.append(torch.as_tensor(batch_camids, dtype=torch.int64))

    if not features:
        raise ValueError("Evaluation dataset is empty")

    return torch.cat(features), torch.cat(pids), torch.cat(camids)


def _average_precision(matches: torch.Tensor) -> float:
    if not matches.any():
        return 0.0

    matches_float = matches.to(torch.float32)
    precision = matches_float.cumsum(0) / torch.arange(1, matches.numel() + 1, dtype=torch.float32)
    return precision[matches].mean().item()


def _compute_metrics(
    distance_matrix: torch.Tensor,
    query_pids: torch.Tensor,
    query_camids: torch.Tensor,
    gallery_pids: torch.Tensor,
    gallery_camids: torch.Tensor,
) -> dict[str, float]:
    cmc_sum = torch.zeros(distance_matrix.shape[1], dtype=torch.float32)
    average_precisions: list[float] = []
    valid_queries = 0

    for index in range(distance_matrix.shape[0]):
        order = torch.argsort(distance_matrix[index])
        ranked_pids = gallery_pids[order]
        ranked_camids = gallery_camids[order]

        keep = ~((ranked_pids == query_pids[index]) & (ranked_camids == query_camids[index]))
        matches = ranked_pids[keep] == query_pids[index]
        if not matches.any():
            continue

        cmc = matches.to(torch.float32).cumsum(0).clamp(max=1)
        cmc_sum[: cmc.numel()] += cmc
        average_precisions.append(_average_precision(matches))
        valid_queries += 1

    if valid_queries == 0:
        return {"mAP": 0.0, "rank-1": 0.0, "rank-5": 0.0, "rank-10": 0.0}

    cmc = cmc_sum / valid_queries
    return {
        "mAP": sum(average_precisions) / valid_queries,
        "rank-1": cmc[0].item(),
        "rank-5": cmc[min(4, cmc.numel() - 1)].item(),
        "rank-10": cmc[min(9, cmc.numel() - 1)].item(),
    }


def evaluate(
    model: ReIDModel,
    query_dir: str | Path,
    gallery_dir: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 4,
) -> dict[str, float]:
    query_loader = _create_eval_loader(query_dir, batch_size=batch_size, num_workers=num_workers)
    gallery_loader = _create_eval_loader(gallery_dir, batch_size=batch_size, num_workers=num_workers)

    query_features, query_pids, query_camids = _extract_embeddings(model, query_loader, device=device)
    gallery_features, gallery_pids, gallery_camids = _extract_embeddings(model, gallery_loader, device=device)

    distance_matrix = torch.cdist(query_features, gallery_features, p=2)
    return _compute_metrics(distance_matrix, query_pids, query_camids, gallery_pids, gallery_camids)
