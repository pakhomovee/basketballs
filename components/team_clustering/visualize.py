"""Team clustering visualization: render videos with team-colored bounding boxes."""

from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from common.classes import PlayersDetections
from common.classes.detections import FrameDetections
from common.utils.utils import get_device
from detector import Detector, get_frame_players_detections
from team_clustering.embedding import PlayerEmbedder, collect_player_crops

TEAM_COLORS = [(0, 0, 255), (255, 0, 0)]  # Red, Blue (BGR)


def _mask_preview(mask: np.ndarray | None, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8), "Mask (none)"
    return (mask > 0.5).astype(np.uint8), "Mask"


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


def plot_masks(sample_detections, num_to_plot=16):
    """Show original / mask / blended triplets for a random subset of detections."""
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

    for i, (crop, mask) in enumerate(samples):
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        binary, mask_title = _mask_preview(mask, crop.shape[:2])

        green = np.zeros_like(crop_rgb)
        green[binary == 1] = [0, 255, 0]
        blended = cv2.addWeighted(crop_rgb, 0.6, green, 0.4, 0)

        idx = i * 3
        axes[idx].imshow(crop_rgb)
        axes[idx].set_title(f"#{i + 1} Original")
        axes[idx].axis("off")

        axes[idx + 1].imshow(binary, cmap="gray")
        axes[idx + 1].set_title(f"#{i + 1} {mask_title}")
        axes[idx + 1].axis("off")

        axes[idx + 2].imshow(blended)
        axes[idx + 2].set_title(f"#{i + 1} Blended")
        axes[idx + 2].axis("off")

    for j in range(len(samples) * 3, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def _render_mask_sample(crop: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    binary, mask_title = _mask_preview(mask, crop.shape[:2])

    green = np.zeros_like(crop_rgb)
    green[binary == 1] = [0, 255, 0]
    blended = cv2.addWeighted(crop_rgb, 0.6, green, 0.4, 0)
    binary_rgb = np.repeat(binary[:, :, None] * 255, 3, axis=2)

    panels = [crop_rgb, binary_rgb, blended]
    height = max(panel.shape[0] for panel in panels)
    resized = []
    for panel in panels:
        if panel.shape[0] != height:
            width = max(1, int(round(panel.shape[1] * height / panel.shape[0])))
            panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_NEAREST)
        resized.append(panel)

    canvas = np.hstack(resized)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas_bgr, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        canvas_bgr,
        mask_title,
        (resized[0].shape[1] + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas_bgr,
        "Blended",
        (resized[0].shape[1] + resized[1].shape[1] + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return canvas_bgr


def browse_masks(sample_detections: list[tuple[np.ndarray, np.ndarray | None]]) -> None:
    """Browse segmentation samples one by one with keyboard controls."""
    if not sample_detections:
        print("No sample detections to plot.")
        return

    index = 0
    total = len(sample_detections)
    window_name = "Segmentation Masks"

    while True:
        crop, mask = sample_detections[index]
        canvas = _render_mask_sample(crop, mask)
        status = f"Sample {index + 1}/{total}  keys: n/right/space next, p/left prev, q/esc quit"
        cv2.putText(canvas, status, (10, canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord(" "), 2555904, 65363):
            index = (index + 1) % total
            continue
        if key in (ord("p"), 2424832, 65361):
            index = (index - 1) % total

    cv2.destroyWindow(window_name)


def collect_mask_samples_from_video(
    video_path: str,
    max_samples: int = 16,
    frame_stride: int = 30,
    detector_conf: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Collect player crops and segmentation masks from a video for quick inspection."""
    if max_samples <= 0:
        return []

    detector = Detector(conf_threshold=detector_conf)
    embedder = PlayerEmbedder(device=get_device())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    samples: list[tuple[np.ndarray, np.ndarray | None]] = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, frame_stride)

    for frame_id in tqdm(range(frame_count), desc="Sampling masks"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % stride != 0:
            continue

        frame_detections = FrameDetections(frame_id, detector.detect_frame(frame))
        players = get_frame_players_detections(frame_detections, conf_threshold=detector_conf)
        if not players:
            continue

        frame_entries = collect_player_crops(frame, players)
        if not frame_entries:
            continue

        boxes = [box for _, box, _ in frame_entries]
        crops = [crop for _, _, crop in frame_entries]
        masks = embedder.get_player_masks_for_frame(frame, boxes)
        samples.extend(zip(crops, masks))
        if len(samples) >= max_samples:
            break

    cap.release()
    return samples[:max_samples]


def visualize_video_masks(
    video_path: str,
    max_samples: int = 16,
    frame_stride: int = 30,
    detector_conf: float = 0.1,
    include_empty: bool = False,
    interactive: bool = True,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Sample detections from a video, segment them, and display their masks."""
    samples = collect_mask_samples_from_video(
        video_path,
        max_samples=max_samples,
        frame_stride=frame_stride,
        detector_conf=detector_conf,
    )
    if not include_empty:
        samples = [(crop, mask) for crop, mask in samples if mask is not None]

    if interactive:
        browse_masks(samples)
    else:
        plot_masks(samples, num_to_plot=max_samples)
    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument("--max-samples", type=int, default=16, help="Number of player crops to visualize")
    parser.add_argument("--frame-stride", type=int, default=30, help="Process every Nth frame")
    parser.add_argument("--detector-conf", type=float, default=0.1, help="Detector confidence threshold")
    parser.add_argument("--include-empty", action="store_true", help="Show crops where segmentation returned no mask")
    parser.add_argument("--grid", action="store_true", help="Show a static grid instead of interactive browsing")
    args = parser.parse_args()

    visualize_video_masks(
        str(Path(args.video_path)),
        max_samples=args.max_samples,
        frame_stride=args.frame_stride,
        detector_conf=args.detector_conf,
        include_empty=args.include_empty,
        interactive=not args.grid,
    )
