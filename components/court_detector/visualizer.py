import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from court_constants import SMALL_COURT_POINTS, FIBA_COURT_POINTS, MAPPING_ROBOFLOW_COURT_DETECTION
import json


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
        default=Path(__file__).parent / "runs" / "detect" / "court_detector" / "court_keypoints_detector" / "weights" / "best.pt",
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

    model = YOLO(str(args.model))
    val_images = load_val_images(args.data_root)
    rng = random.Random()

    num_classes = int(max(p[2] for p in SMALL_COURT_POINTS)) + 1
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
        label_path = args.data_root / "labels" / "val" / (img_path.stem + ".txt")
        gt_pts = load_gt_points(label_path, w, h)

        # Predictions
        preds = model.predict(img_rgb, verbose=False)[0]
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
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor=cmap(cls_id), alpha=0.8),
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
    plt.show()


def run_deepsportradar(args):
    ds_root = Path(__file__).parent / "dataset" / "deepsportradar_instants_dataset"
    json_files = list(ds_root.glob("*/*/*.json"))
    if not json_files:
        raise RuntimeError(f"No json files found under {ds_root}")

    num_classes = int(max(p[2] for p in SMALL_COURT_POINTS)) + 1
    cmap = plt.colormaps.get_cmap("tab20")
    rng = random.Random()
    fig, ax = plt.subplots(figsize=(9, 5))

    def project_points(K, R, T, pts):
        pts_cam = (R @ pts.T + T.reshape(3, 1))
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
            pts.append([(x + 14.0) * 100, (y + 7.5) * 100, 0.0])  # shift to dataset origin (left boundary) and scales from meters to cantimeters
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
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor=cmap(cls_id), alpha=0.8),
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
    plt.show()


def run_sportcenter(args):
    ds_root = Path(__file__).parent / "dataset" / "sportcenter_camerapose_dataset"
    seqs = [d for d in ds_root.iterdir() if d.is_dir() and d.name.startswith("seq_")]
    if not seqs:
        raise RuntimeError("No sportcenter sequences found.")
    rng = random.Random()
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.colormaps.get_cmap("tab20")
    num_classes = int(max(p[2] for p in SMALL_COURT_POINTS)) + 1

    def project_homography(points_xy, H):
        p = np.vstack([points_xy.T, np.ones(len(points_xy))])
        transformed = H @ p
        return (transformed[:2] / transformed[2]).T

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

        pts_world = np.array([(y, x, t) for x, y, t in SMALL_COURT_POINTS], dtype=np.float32)
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
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor=cmap(cls_id % 20), alpha=0.8),
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
    plt.show()


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
        kpts = np.array(vals[5:5 + kpt_num * 3], dtype=float).reshape(kpt_num, 3)
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
                                boxstyle="round,pad=0.1", facecolor="white", edgecolor=cmap(cls_id % 20), alpha=0.8
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
    plt.show()


if __name__ == "__main__":
    main()
