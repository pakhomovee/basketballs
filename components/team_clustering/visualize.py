"""Team clustering visualization: render videos with team-colored bounding boxes."""

from __future__ import annotations

import argparse
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from common.classes import PlayersDetections
from common.utils.utils import get_device
from team_clustering.embedding import PlayerEmbedder
from team_clustering.shared import DEFAULT_SEG_MODEL, clip_bbox, resolve_repo_path

TEAM_COLORS = [(0, 0, 255), (255, 0, 0)]  # Red, Blue (BGR)
MASK_COLOR = np.array([0, 255, 0], dtype=np.uint8)
BBOX_COLOR = (255, 0, 0)
DEFAULT_FRAME_STRIDE = 30
DEFAULT_NUM_MASKS = 16
DEFAULT_MAX_DETECTIONS_PER_FRAME = 4
MaskSample = tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, np.ndarray]


def render_clustered_video(video_path: str, detections: PlayersDetections, output_path: str):
    """Write a new video with bounding boxes coloured by team assignment."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_id = 0

    for _ in tqdm(range(total), desc="Writing video"):
        ret, frame = cap.read()
        if not ret:
            break
        for player in detections.get(frame_id, []):
            if player.team_id is not None:
                color = TEAM_COLORS[player.team_id % len(TEAM_COLORS)]
                x1, y1, x2, y2 = map(int, player.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID:{player.player_id} T:{player.team_id + 1}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"Saved clustered video to {output_path}")


def plot_masks(
    sample_detections: list[MaskSample],
    num_to_plot: int = DEFAULT_NUM_MASKS,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Show full-frame bbox context, crop, and blended crop for sampled detections."""
    if not sample_detections:
        print("No sample detections to plot.")
        return

    samples = sample_detections
    if len(samples) > num_to_plot:
        samples = random.sample(samples, num_to_plot)

    rows = len(samples)
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.flatten()

    for i, (frame, bbox, crop, mask) in enumerate(samples):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        binary = (mask > 0.5).astype(np.uint8)

        x1, y1, x2, y2 = bbox
        frame_with_bbox = frame_rgb.copy()
        cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), BBOX_COLOR, 2)

        green = np.zeros_like(crop_rgb)
        green[binary == 1] = MASK_COLOR
        blended = cv2.addWeighted(crop_rgb, 0.6, green, 0.4, 0)

        idx = i * 3
        axes[idx].imshow(frame_with_bbox)
        axes[idx].set_title(f"#{i + 1} Frame + BBox")
        axes[idx].axis("off")

        axes[idx + 1].imshow(crop_rgb)
        axes[idx + 1].set_title(f"#{i + 1} Crop")
        axes[idx + 1].axis("off")

        axes[idx + 2].imshow(blended)
        axes[idx + 2].set_title(f"#{i + 1} Crop + Mask")
        axes[idx + 2].axis("off")

    for j in range(len(samples) * 3, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved mask gallery to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _sample_mask_detections_from_frame(
    frame: np.ndarray,
    result,
    max_detections: int,
    embedder: PlayerEmbedder,
) -> list[MaskSample]:
    if not result.boxes or not result.masks or not result.masks.xy:
        return []

    frame_h, frame_w = frame.shape[:2]
    detections = []
    for box_tensor, poly in zip(result.boxes.xyxy.cpu().numpy(), result.masks.xy, strict=False):
        clipped = clip_bbox(tuple(box_tensor.tolist()), frame_w, frame_h)
        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        polygon = np.array(poly).reshape((-1, 1, 2)).astype(int)
        cv2.fillPoly(full_mask, [polygon], 1)
        raw_mask_crop = full_mask[y1:y2, x1:x2]
        refined_mask_crop = embedder.refine_embedding_mask(crop, raw_mask_crop)
        detections.append((frame.copy(), clipped, crop, refined_mask_crop, int(refined_mask_crop.sum())))

    detections.sort(key=lambda item: item[4], reverse=True)
    return [(full_frame, bbox, crop, mask) for full_frame, bbox, crop, mask, _ in detections[:max_detections]]


def collect_video_mask_samples(
    video_path: str,
    seg_model: str = str(DEFAULT_SEG_MODEL),
    frame_stride: int = 30,
    max_samples: int = 16,
    max_detections_per_frame: int = 4,
) -> list[MaskSample]:
    """Sample player masks from a video by running person segmentation on every Nth frame."""
    video_path = str(resolve_repo_path(video_path))
    seg_model = str(resolve_repo_path(seg_model))
    embedder = PlayerEmbedder(seg_model, get_device())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_detections: list[MaskSample] = []

    for frame_idx in tqdm(range(total_frames), desc="Sampling video masks"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, frame_stride) != 0:
            continue

        result = embedder.model(frame, verbose=False, classes=[embedder.player_class_id], device=embedder.device)[0]
        sample_detections.extend(_sample_mask_detections_from_frame(frame, result, max_detections_per_frame, embedder))
        if len(sample_detections) >= max_samples:
            break

    cap.release()
    return sample_detections[:max_samples]


def visualize_video_masks(
    video_path: str,
    seg_model: str = str(DEFAULT_SEG_MODEL),
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_samples: int = DEFAULT_NUM_MASKS,
    max_detections_per_frame: int = DEFAULT_MAX_DETECTIONS_PER_FRAME,
    show: bool = True,
    save_path: str | None = None,
) -> list[MaskSample]:
    sample_detections = collect_video_mask_samples(
        video_path=video_path,
        seg_model=seg_model,
        frame_stride=frame_stride,
        max_samples=max_samples,
        max_detections_per_frame=max_detections_per_frame,
    )
    plot_masks(sample_detections, num_to_plot=max_samples, show=show, save_path=save_path)
    return sample_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sample person masks from a video")
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument("--seg-model", default=str(DEFAULT_SEG_MODEL), help="Segmentation model checkpoint")
    args = parser.parse_args()

    visualize_video_masks(
        video_path=args.video_path,
        seg_model=args.seg_model,
        frame_stride=DEFAULT_FRAME_STRIDE,
        max_samples=DEFAULT_NUM_MASKS,
        max_detections_per_frame=DEFAULT_MAX_DETECTIONS_PER_FRAME,
    )


if __name__ == "__main__":
    main()
