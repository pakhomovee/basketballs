"""ReID evaluation: mAP and CMC rank-k accuracy."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import QueryGalleryDataset


@torch.no_grad()
def extract_all_features(
    model: torch.nn.Module,
    dataset: QueryGalleryDataset,
    batch_size: int = 128,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (features, pids, camids) numpy arrays for a query/gallery set."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_feats, all_pids, all_cams = [], [], []
    for imgs, pids, cams in loader:
        feats = model(imgs.to(device))
        all_feats.append(feats.cpu().numpy())
        all_pids.append(np.array(pids))
        all_cams.append(np.array(cams))

    return np.concatenate(all_feats), np.concatenate(all_pids), np.concatenate(all_cams)


def compute_distance_matrix(query_feats: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between query and gallery features."""
    # Features are already L2-normalised, so euclidean distance ∝ (2 - 2·cosine)
    qf = torch.from_numpy(query_feats)
    gf = torch.from_numpy(gallery_feats)
    dist = torch.cdist(qf, gf, p=2).numpy()
    return dist


def eval_map_cmc(
    dist_matrix: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_cams: np.ndarray,
    gallery_cams: np.ndarray,
    max_rank: int = 10,
) -> tuple[float, np.ndarray]:
    """Compute mAP and CMC curve.

    SynergyReID: query = frame 0, gallery = frames 1–19. Correct matches are
    same pid (same sequence). No junk exclusion needed.
    """
    num_q = dist_matrix.shape[0]
    indices = np.argsort(dist_matrix, axis=1)

    all_ap = []
    all_cmc = np.zeros(max_rank, dtype=np.float32)

    for i in range(num_q):
        q_pid = query_pids[i]

        order = indices[i]
        g_pids = gallery_pids[order]

        matches = (g_pids == q_pid).astype(np.float32)
        if matches.sum() == 0:
            continue

        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc += cmc[:max_rank]

        # AP
        num_relevant = matches.sum()
        cum_correct = matches.cumsum()
        precision_at_k = cum_correct / np.arange(1, len(matches) + 1)
        ap = (precision_at_k * matches).sum() / num_relevant
        all_ap.append(ap)

    mAP = float(np.mean(all_ap)) if all_ap else 0.0
    cmc_curve = all_cmc / num_q
    return mAP, cmc_curve


def evaluate(
    model: torch.nn.Module,
    query_dir: str,
    gallery_dir: str,
    device: str = "cuda",
    batch_size: int = 128,
) -> dict[str, float]:
    """Full evaluation: extract features → compute distances → mAP + CMC."""
    q_ds = QueryGalleryDataset(query_dir)
    g_ds = QueryGalleryDataset(gallery_dir)

    q_feats, q_pids, q_cams = extract_all_features(model, q_ds, batch_size, device)
    g_feats, g_pids, g_cams = extract_all_features(model, g_ds, batch_size, device)

    dist = compute_distance_matrix(q_feats, g_feats)
    mAP, cmc = eval_map_cmc(dist, q_pids, g_pids, q_cams, g_cams)

    return {
        "mAP": mAP,
        "rank-1": float(cmc[0]),
        "rank-5": float(cmc[4]) if len(cmc) > 4 else 0.0,
        "rank-10": float(cmc[9]) if len(cmc) > 9 else 0.0,
    }
