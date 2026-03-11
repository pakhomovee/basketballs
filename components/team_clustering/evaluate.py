"""
Evaluate team clustering on ground truth detections.

Loads GT bboxes, runs team clustering (embedding + KMeans), compares to GT labels.

Usage:
    python -m components.team_clustering.evaluate

Metrics: accuracy (per-detection), adjusted rand index.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

from common.classes.player import Player
from team_clustering.embedding import PlayerEmbedder, extract_embeddings_from_image
from team_clustering.team_clustering import TeamClustering


def evaluate(ground_truth_path: str | Path) -> dict[str, float]:
    """
    Run team clustering on GT bboxes and compare to GT labels.

    Args:
        ground_truth_path: JSON with list of {image_path, bboxes, team_labels}.

    Returns:
        dict with accuracy, adjusted_rand_index, total_detections, correct.
    """
    with open(ground_truth_path) as f:
        gt_list = json.load(f)

    _root = Path(__file__).resolve().parent.parent.parent

    embedder = PlayerEmbedder(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    tc = TeamClustering(n_clusters=2)

    correct = 0
    total = 0
    all_pred_aligned: list[int] = []
    all_gt: list[int] = []

    for g in tqdm(gt_list, desc="Evaluate"):
        path = g["image_path"]
        gt_bboxes = g["bboxes"]
        gt_labels = g["team_labels"]

        img_path = _root / path if not Path(path).is_absolute() else Path(path)
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        players = [Player(bbox=bbox, player_id=i) for i, bbox in enumerate(gt_bboxes)]
        if len(players) < 2:
            continue

        extract_embeddings_from_image(frame, players, embedder=embedder)
        clusters = tc.cluster_per_frame(players)

        pred_labels = []
        gt_labels_valid = []
        for i, gl in enumerate(gt_labels):
            if gl is None or gl == "any":
                continue
            if i not in clusters:
                continue
            pred_labels.append(clusters[i])
            gt_labels_valid.append(gl)

        if not pred_labels:
            continue

        # Cluster labels 0/1 are arbitrary per image: pick best mapping per image
        pred_arr = np.array(pred_labels)
        gt_arr_img = np.array(gt_labels_valid)
        correct_identity = int(np.sum(pred_arr == gt_arr_img))
        correct_swap = int(np.sum(pred_arr == 1 - gt_arr_img))
        correct += max(correct_identity, correct_swap)
        total += len(pred_labels)

        # Align pred to GT for ARI (apply best mapping)
        if correct_swap > correct_identity:
            pred_arr = 1 - pred_arr
        all_pred_aligned.extend(pred_arr.tolist())
        all_gt.extend(gt_labels_valid)

    accuracy = correct / total if total > 0 else 0.0

    pred_arr = np.array(all_pred_aligned)
    gt_arr = np.array(all_gt)
    ari = adjusted_rand_score(gt_arr, pred_arr) if len(pred_arr) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "adjusted_rand_index": ari,
        "total_detections": total,
        "correct": correct,
    }


def main() -> None:
    _root = Path(__file__).resolve().parent.parent.parent
    ground_truth_path = _root / "dataset" / "team_clustering_benchmark" / "ground_truth.json"

    results = evaluate(ground_truth_path)

    print("Team clustering evaluation")
    print("=" * 40)
    print(f"  Accuracy:            {results['accuracy']:.4f}")
    print(f"  Adjusted Rand Index: {results['adjusted_rand_index']:.4f}")
    print(f"  Correct / Total:    {results['correct']} / {results['total_detections']}")


if __name__ == "__main__":
    main()
