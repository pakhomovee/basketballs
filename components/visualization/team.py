"""Team clustering visualization: render videos with team-colored bounding boxes."""

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from common.classes import FrameDetections

TEAM_COLORS = [(0, 0, 255), (255, 0, 0)]  # Red, Blue (BGR)


def render_clustered_video(video_path: str, detections: FrameDetections, output_path: str):
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
        binary = (mask > 0.5).astype(np.uint8)

        green = np.zeros_like(crop_rgb)
        green[binary == 1] = [0, 255, 0]
        blended = cv2.addWeighted(crop_rgb, 0.6, green, 0.4, 0)

        idx = i * 3
        axes[idx].imshow(crop_rgb)
        axes[idx].set_title(f"#{i + 1} Original")
        axes[idx].axis("off")

        axes[idx + 1].imshow(binary, cmap="gray")
        axes[idx + 1].set_title(f"#{i + 1} Mask")
        axes[idx + 1].axis("off")

        axes[idx + 2].imshow(blended)
        axes[idx + 2].set_title(f"#{i + 1} Blended")
        axes[idx + 2].axis("off")

    for j in range(len(samples) * 3, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
