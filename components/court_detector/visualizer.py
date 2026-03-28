import argparse
import os
import random
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

from common.classes import CourtType
from court_detector.court_constants import (
    SMALL_COURT_POINTS,
    FIBA_COURT_POINTS,
    MAPPING_ROBOFLOW_COURT_DETECTION,
    CourtConstants,
)
from court_detector.court_detector import CourtDetector, project_homography
from court_detector.prepare_dataset import SPORCENTER_GOOD_SEQS
from config import load_default_config
from video_reader import VideoReader
import json
import matplotlib.pyplot as plt


def _show_plot():
    backend = matplotlib.get_backend().lower()
    if backend.endswith("agg") or backend in {"agg", "pdf", "svg", "ps"}:
        out_path = Path(tempfile.gettempdir()) / f"court_vis_{int(time.time() * 1000)}.png"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved plot to {out_path} (non-interactive backend: {backend})")
        return
    _show_plot()


def load_val_images(data_root: Path):
    val_dir = data_root / "images" / "val"
    imgs = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        raise RuntimeError(f"No val images found in {val_dir}")
    return imgs


def load_gt_points(label_path: Path, w: int, h: int):
    if not label_path.is_file():
        return np.empty((0, 3))
    pts = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            pts.append((cx * w, cy * h, cls))
    return np.array(pts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs predicted court points on val images.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent.parent.parent / "models" / "court_detection_model.pt",
        help="Path to trained YOLOv8 model",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parent / "yolo_combined_data",
        help="Path to YOLO detection dataset (with images/val and labels/val)",
    )
    parser.add_argument(
        "--deepsportradar",
        action="store_true",
        help="Show random frame from deepsportradar_instants_dataset with projected court points",
    )
    parser.add_argument(
        "--sportcenter",
        action="store_true",
        help="Show random frame from sportcenter_camerapose_dataset with projected SMALL_COURT_POINTS",
    )
    parser.add_argument(
        "--roboflow-court-detection",
        action="store_true",
        help="Show random frame from roboflow_court_detection (COCO pose) with GT keypoints",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Run inference on frames from a video file",
    )
    args = parser.parse_args()

    if args.deepsportradar:
        run_deepsportradar(args)
        return
    if args.sportcenter:
        run_sportcenter(args)
        return
    if args.roboflow_court_detection:
        run_roboflow(args)
        return
    if args.video:
        run_video(args)
        return

    model = YOLO(str(args.model))
    val_images = load_val_images(args.data_root)
    rng = random.Random()
    court_detection_threshold = load_default_config().court_detector.detection_threshold

    # num_classes = int(max(p[2] for p in SMALL_COURT_POINTS)) + 1
    cmap = plt.colormaps.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(9, 5))

    def show():
        img_path = rng.choice(val_images)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # GT points from labels
        # label_path = args.data_root / "labels" / "val" / (img_path.stem + ".txt")
        # gt_pts = load_gt_points(label_path, w, h)

        # Predictions
        preds = model.predict(img_rgb, verbose=False, conf=court_detection_threshold)[0]
        pred_boxes = preds.boxes.xywh.cpu().numpy() if preds.boxes is not None else np.empty((0, 4))
        pred_cls = preds.boxes.cls.cpu().numpy().astype(int) if preds.boxes is not None else np.empty((0,), dtype=int)

        ax.clear()
        ax.imshow(img_rgb)

        # draw GT
        # for cls_id in np.unique(gt_pts[:, 2]) if len(gt_pts) else []:
        #     mask = gt_pts[:, 2] == cls_id
        #     ax.scatter(
        #         gt_pts[mask, 0],
        #         gt_pts[mask, 1],
        #         color=cmap(cls_id),
        #         marker="x",
        #         s=70,
        #         label=f"gt type {cls_id}",
        #     )

        # draw predictions
        for cls_id in np.unique(pred_cls) if len(pred_cls) else []:
            mask = pred_cls == cls_id
            centers = pred_boxes[mask, :2]
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                facecolors="none",
                edgecolors=cmap(cls_id),
                s=90,
                marker="o",
                linewidths=2,
                label=f"pred type {cls_id}",
            )
            # label centers
            for x, y in centers:
                ax.text(
                    x + 8,
                    y - 8,
                    str(cls_id),
                    color=cmap(cls_id),
                    fontsize=12,
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        facecolor="white",
                        edgecolor=cmap(cls_id),
                        alpha=0.8,
                    ),
                )

        ax.set_title(img_path.name)
        ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    _show_plot()


def run_deepsportradar(args):
    ds_root = Path(__file__).parent / "dataset" / "deepsportradar_instants_dataset"
    json_files = list(ds_root.glob("*/*/*.json"))
    if not json_files:
        raise RuntimeError(f"No json files found under {ds_root}")

    # num_classes = int(max(p[2] for p in SMALL_COURT_POINTS)) + 1
    cmap = plt.colormaps.get_cmap("tab20")
    rng = random.Random()
    fig, ax = plt.subplots(figsize=(9, 5))

    def project_points(K, R, T, pts):
        pts_cam = R @ pts.T + T.reshape(3, 1)
        uvw = K @ pts_cam
        return (uvw[:2] / uvw[2]).T

    def show():
        jpath = rng.choice(json_files)
        with open(jpath, "r") as f:
            meta = json.load(f)
        cal = meta["calibration"]
        K = np.array(cal["KK"], dtype=np.float32).reshape(3, 3)
        R = np.array(cal["R"], dtype=np.float32).reshape(3, 3)
        T = np.array(cal["T"], dtype=np.float32)

        # try frame *_0.png else .png
        base = jpath.with_suffix("")
        png_path = Path(str(base) + "_0.png")
        if not png_path.is_file():
            png_path = Path(str(base) + ".png")
        img_bgr = cv2.imread(str(png_path))
        if img_bgr is None:
            return
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pts = []
        types = []
        for x, y, t in FIBA_COURT_POINTS:
            pts.append(
                [(x + 14.0) * 100, (-y + 7.5) * 100, 0.0]
            )  # shift to dataset origin (left boundary) and scales from meters to cantimeters
            types.append(t)
        pts = np.array(pts, dtype=np.float32)
        types = np.array(types, dtype=int)

        pts_img = project_points(K, R, T, pts)
        inside = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
        pts_img, types = pts_img[inside], types[inside]

        ax.clear()
        ax.imshow(img_rgb)
        for cls_id in np.unique(types) if len(types) else []:
            mask = types == cls_id
            ax.scatter(
                pts_img[mask, 0],
                pts_img[mask, 1],
                color=cmap(cls_id),
                s=60,
                marker="x",
                label=f"type {cls_id}",
            )
            for x, y in pts_img[mask]:
                ax.text(
                    x + 8,
                    y - 8,
                    str(cls_id),
                    color=cmap(cls_id),
                    fontsize=12,
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        facecolor="white",
                        edgecolor=cmap(cls_id),
                        alpha=0.8,
                    ),
                )
        ax.set_title(jpath.name)
        ax.axis("off")
        if len(types):
            ax.legend()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    _show_plot()


def run_sportcenter(args):
    ds_root = Path(__file__).parent / "dataset" / "sportcenter_camerapose_dataset"
    seqs = [ds_root / f"seq_{seq_id}" for seq_id in SPORCENTER_GOOD_SEQS]
    if not seqs:
        raise RuntimeError("No sportcenter sequences found.")
    rng = random.Random()
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.colormaps.get_cmap("tab20")
    # num_classes = int(max(p[2] for p in SMALL_COURT_POINTS)) + 1

    def show():
        seq = rng.choice(seqs)
        poses_path = seq / "poses.json"
        with open(poses_path, "r") as f:
            poses = json.load(f)
        filename = rng.choice(list(poses.keys()))
        Hr = np.array(poses[filename]["Hr"], dtype=np.float32)
        img_path = seq / "images_orig_blurred" / Path(filename).name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pts_world = np.array([(y, -x, t) for x, y, t in SMALL_COURT_POINTS], dtype=np.float32)
        pts_xy = pts_world[:, :2]
        pts_types = pts_world[:, 2].astype(int)
        pts_img = project_homography(pts_xy, Hr)

        inside = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
        pts_img, pts_types = pts_img[inside], pts_types[inside]

        ax.clear()
        ax.imshow(img_rgb)
        for cls_id in np.unique(pts_types) if len(pts_types) else []:
            mask = pts_types == cls_id
            ax.scatter(
                pts_img[mask, 0],
                pts_img[mask, 1],
                color=cmap(cls_id % 20),
                s=60,
                marker="x",
                label=f"type {cls_id}",
            )
            for x, y in pts_img[mask]:
                ax.text(
                    x + 8,
                    y - 8,
                    str(cls_id),
                    color=cmap(cls_id % 20),
                    fontsize=12,
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        facecolor="white",
                        edgecolor=cmap(cls_id % 20),
                        alpha=0.8,
                    ),
                )
        ax.set_title(f"{seq.name}/{filename}")
        ax.axis("off")
        if len(pts_types):
            ax.legend()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    _show_plot()


def run_roboflow(args):
    ds_root = Path(__file__).parent / "dataset" / "roboflow_court_detection"
    val_dir = ds_root / "valid"
    img_dir = val_dir / "images"
    lbl_dir = val_dir / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise RuntimeError(f"No images in {img_dir}")
    kpt_num = 33  # from data.yaml kpt_shape
    rng = random.Random()
    cmap = plt.colormaps.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 5))

    def load_label(label_path):
        if not label_path.is_file():
            return None
        with open(label_path) as f:
            line = f.readline().strip()
        if not line:
            return None
        vals = list(map(float, line.split()))
        if len(vals) < 5 + kpt_num * 3:
            return None
        cls = int(vals[0])
        bbox = vals[1:5]
        kpts = np.array(vals[5 : 5 + kpt_num * 3], dtype=float).reshape(kpt_num, 3)
        return cls, bbox, kpts

    def show():
        img_path = rng.choice(images)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        label_path = lbl_dir / (img_path.stem + ".txt")
        parsed = load_label(label_path)
        ax.clear()
        ax.imshow(img_rgb)
        if parsed is not None:
            _, _, kpts = parsed
            xs = kpts[:, 0] * w
            ys = kpts[:, 1] * h
            vis = kpts[:, 2] > 0
            vis_idx = np.where(vis)[0]
            remapped = []
            for i in vis_idx:
                if i in MAPPING_ROBOFLOW_COURT_DETECTION:
                    remapped.append((xs[i], ys[i], MAPPING_ROBOFLOW_COURT_DETECTION[i]))
            if remapped:
                remapped = np.array(remapped)
                for cls_id in np.unique(remapped[:, 2]).astype(int):
                    mask = remapped[:, 2] == cls_id
                    ax.scatter(
                        remapped[mask, 0],
                        remapped[mask, 1],
                        color=cmap(cls_id % 20),
                        s=40,
                        marker="x",
                        label=f"type {cls_id}",
                    )
                    for x, y in remapped[mask][:, :2]:
                        ax.text(
                            x + 8,
                            y - 8,
                            str(cls_id),
                            color=cmap(cls_id % 20),
                            fontsize=12,
                            weight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.1",
                                facecolor="white",
                                edgecolor=cmap(cls_id % 20),
                                alpha=0.8,
                            ),
                        )
        ax.set_title(img_path.name)
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    _show_plot()


def run_video(args):
    # Detector and court constants
    detector = CourtDetector(str(args.model), cfg=load_default_config())
    court_constants = CourtConstants(CourtType.NBA)

    # Pre-compute homographies, keypoint detections and per-frame losses
    # homographies, frames_sizes, keypoints_detections, losses = detector.extract_homographies_from_video(
    #     str(args.video), court_constants
    # )

    vr = VideoReader(str(args.video))
    homographies, frames_sizes, keypoints_detections, losses = detector.extract_homographies_from_video_v2(
        vr, court_constants
    )
    # losses = None

    # Rewind for visualization
    vr.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Load court model image (NBA court)
    court_img_path = Path(__file__).parent / "nba.png"
    court_bgr = cv2.imread(str(court_img_path))
    if court_bgr is None:
        raise RuntimeError(f"Cannot read court image: {court_img_path}")
    ch, cw = court_bgr.shape[:2]

    alpha = 0.5  # transparency for court overlay
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]

    frame_idx = 0
    window_name = f"Court visualization - {args.video.name}"

    while True:
        # Clamp frame index to valid range
        if frame_idx < 0:
            frame_idx = 0
        if frame_idx >= len(homographies):
            frame_idx = len(homographies) - 1

        # Seek to the desired frame and read it
        vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = vr.read()
        if not ret:
            break

        H = homographies[frame_idx]
        frame_w, frame_h = frames_sizes[frame_idx]
        pred_centers, pred_cls = keypoints_detections[frame_idx]

        # Prepare base frame
        frame_vis = frame_bgr.copy()

        if H is not None:
            # Map court image corners (center at (0,0), edges at +/-0.5) to frame using inverse homography
            H_inv = np.linalg.inv(H)
            court_corners_norm = np.array(
                [
                    [-0.5, -0.5],  # left-bottom
                    [0.5, -0.5],  # right-bottom
                    [0.5, 0.5],  # right-top
                    [-0.5, 0.5],  # left-top
                ],
                dtype=np.float32,
            )
            frame_corners_norm = project_homography(court_corners_norm, H_inv)
            dst_pts = frame_corners_norm * np.array([[frame_w, frame_h]], dtype=np.float32)

            src_pts = np.array(
                [
                    [0.0, 0.0],
                    [float(cw), 0.0],
                    [float(cw), float(ch)],
                    [0.0, float(ch)],
                ],
                dtype=np.float32,
            )

            M = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))
            warped_court = cv2.warpPerspective(court_bgr, M, (frame_w, frame_h))

            # Alpha blend court and frame (both in BGR)
            frame_vis = (alpha * warped_court + (1.0 - alpha) * frame_vis).astype(np.uint8)

        # Draw keypoint detections (small circles with class labels)
        if pred_centers is not None and len(pred_centers):
            for (x, y), cls_id in zip(pred_centers, pred_cls):
                c = colors[int(cls_id) % len(colors)]
                center = (int(round(x)), int(round(y)))
                cv2.circle(frame_vis, center, 4, c, -1)
                cv2.putText(
                    frame_vis,
                    str(int(cls_id)),
                    (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    c,
                    1,
                    cv2.LINE_AA,
                )

        # Show frame and print current loss
        cv2.imshow(window_name, frame_vis)
        if losses is not None and len(losses) > frame_idx:
            print(f"Frame {frame_idx}: loss = {float(losses[frame_idx]):.4f}")

        # Left/right arrows to move backward/forward, q/ESC to exit
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):
            break
        # OpenCV arrow key codes: left=81, up=82, right=83, down=84 (on most platforms)
        if key == 81:  # left arrow
            frame_idx -= 1
        elif key == 83:  # right arrow
            frame_idx += 1
        else:
            # Any other key: stay on the same frame
            continue

    vr.release()
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
