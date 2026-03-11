"""
Visualize team clustering predictions vs ground truth for debugging.

Saves side-by-side images: GT labels (left) vs Pred labels (right).
Green border = correct, red border = wrong (after best per-image mapping).

Usage:
    python -m components.team_clustering.visualize_eval [--max-images N] [--output-dir PATH]
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from common.classes.player import Player
from team_clustering.embedding import PlayerEmbedder, extract_embeddings_from_image
from team_clustering.team_clustering import TeamClustering

# BGR: team 0 = blue, team 1 = red
TEAM_COLORS = [(255, 100, 50), (50, 50, 255)]
CORRECT_COLOR = (0, 255, 0)   # green
WRONG_COLOR = (0, 0, 255)     # red


def draw_boxes(frame: np.ndarray, bboxes: list, labels: list, border_colors: list[int] | None = None) -> np.ndarray:
    """Draw bboxes with team colors. border_colors overrides if provided (for correct/wrong)."""
    out = frame.copy()
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = map(int, bbox[:4])
        if border_colors is not None:
            color = (CORRECT_COLOR if border_colors[i] else WRONG_COLOR)
        else:
            color = TEAM_COLORS[label] if label in (0, 1) else (128, 128, 128)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Visualize team clustering vs GT")
    _root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--max-images", type=int, default=10, help="Max images to visualize")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_root / "dataset" / "team_clustering_benchmark" / "visualizations",
        help="Output directory",
    )
    args = parser.parse_args()

    gt_path = _root / "dataset" / "team_clustering_benchmark" / "ground_truth.json"
    with open(gt_path) as f:
        gt_list = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    embedder = PlayerEmbedder(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    tc = TeamClustering(n_clusters=2)

    for idx, g in enumerate(gt_list[: args.max_images]):
        path = g["image_path"]
        gt_bboxes = g["bboxes"]
        gt_labels = g["team_labels"]

        img_path = _root / path if not Path(path).is_absolute() else Path(path)
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Skipping {path} (cannot load)")
            continue

        players = [Player(bbox=bbox, player_id=i) for i, bbox in enumerate(gt_bboxes)]
        if len(players) < 2:
            continue

        extract_embeddings_from_image(frame, players, embedder=embedder)
        clusters = tc.cluster_per_frame(players)

        pred_labels = []
        gt_valid = []
        valid_indices = []
        for i, gl in enumerate(gt_labels):
            if gl is None or gl == "any":
                continue
            if i not in clusters:
                continue
            pred_labels.append(clusters[i])
            gt_valid.append(gl)
            valid_indices.append(i)

        if not pred_labels:
            continue

        pred_arr = np.array(pred_labels)
        gt_arr = np.array(gt_valid)
        correct_identity = int(np.sum(pred_arr == gt_arr))
        correct_swap = int(np.sum(pred_arr == 1 - gt_arr))

        use_swap = correct_swap > correct_identity
        if use_swap:
            pred_arr = 1 - pred_arr

        correct_mask = pred_arr == gt_arr

        # Build pred/gt label lists for ALL bboxes (for drawing)
        pred_full = [None] * len(gt_bboxes)
        gt_full = [None] * len(gt_bboxes)
        border_colors_full = [None] * len(gt_bboxes)
        for k, vi in enumerate(valid_indices):
            pred_full[vi] = int(pred_arr[k])
            gt_full[vi] = gt_valid[k]
            border_colors_full[vi] = correct_mask[k]

        # Filter to valid for drawing
        draw_bboxes = [gt_bboxes[i] for i in valid_indices]
        draw_gt = [gt_full[i] for i in valid_indices]
        draw_pred = [pred_full[i] for i in valid_indices]
        draw_correct = [border_colors_full[i] for i in valid_indices]

        img_gt = draw_boxes(frame, draw_bboxes, draw_gt)
        img_pred = draw_boxes(frame, draw_bboxes, draw_pred, border_colors=draw_correct)

        h, w = frame.shape[:2]
        max_h = 600
        if h > max_h:
            scale = max_h / h
            img_gt = cv2.resize(img_gt, (int(w * scale), int(h * scale)))
            img_pred = cv2.resize(img_pred, (int(w * scale), int(h * scale)))

        h_disp, w_disp = img_gt.shape[:2]
        combined = np.hstack([img_gt, img_pred])
        cv2.putText(combined, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Pred (green=correct, red=wrong)", (w_disp + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        n_correct = int(np.sum(correct_mask))
        n_total = len(correct_mask)
        acc = n_correct / n_total if n_total > 0 else 0
        cv2.putText(combined, f"Acc: {n_correct}/{n_total} = {acc:.2f}", (10, h_disp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = args.output_dir / f"{idx:03d}_{Path(path).stem[:50]}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved {out_path}")

    print(f"\nSaved {args.max_images} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
