"""
Prepare a unified detection dataset from multiple sources:
- sportcenter_camerapose_dataset
- deepsportradar_instants_dataset
- roboflow_court_detection (COCO keypoints in YOLO txt format)

Output: one YOLO detection dataset under components/court_detector/yolo_combined_data
with images/train|val and labels/train|val. Each label file ends with an extra line
containing the dataset name (sportcenter, deepsportradar, roboflow-court-detection)
for downstream sampling.
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from court_detector.court_constants import (
    SMALL_COURT_POINTS,
    FIBA_COURT_POINTS,
    MAPPING_ROBOFLOW_COURT_DETECTION,
    NBA_COURT_POINTS,
)

JPEG_QUALITY = 50
SPORTCENTER_FRACTION = 0.2
BOX_SZ = 0.02  # fraction of the longer image side

SPORCENTER_GOOD_SEQS = [
    9841,
    9844,
    9845,
    9850,
    9851,
    9852,
    9853,
    171305,
    171931,
    172146,
    172318,
    172444,
    172647,
    173321,
    173510,
    173742,
    173833,
    174006,
    174210,
]


def ensure_dirs(base: Path):
    dirs = {
        "img_train": base / "images" / "train",
        "img_val": base / "images" / "val",
        "lbl_train": base / "labels" / "train",
        "lbl_val": base / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_label(path: Path, lines: list[str], dataset_tag: str):
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")
        f.write(dataset_tag)


def project_homography(points_xy, H):
    p = np.vstack([points_xy.T, np.ones(len(points_xy))])
    transformed = H @ p
    return (transformed[:2] / transformed[2]).T


def save_jpeg(dst_path: Path, img: np.ndarray):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"Failed to write image: {dst_path}")


def convert_sportcenter(out_dirs, val_split=0.15):
    dataset_root = Path(__file__).parent / "dataset" / "sportcenter_camerapose_dataset"
    seqs = [dataset_root / f"seq_{seq_id}" for seq_id in SPORCENTER_GOOD_SEQS]
    if not seqs:
        return
    val_count = max(1, int(len(seqs) * val_split))
    val_seqs = set(s.name for s in seqs[:val_count])
    # train_seqs = set(s.name for s in seqs[val_count:])

    samples = []
    for seq in seqs:
        split = "val" if seq.name in val_seqs else "train"
        poses_path = seq / "poses.json"
        if not poses_path.is_file():
            continue
        poses = json.load(open(poses_path))
        for fname, pose in poses.items():
            Hr = np.array(pose["Hr"], dtype=np.float32)
            img_path = seq / "images_orig_blurred" / Path(fname).name
            if img_path.is_file():
                samples.append((img_path, Hr, split))
    if SPORTCENTER_FRACTION < 1.0:
        rng = random.Random(42)
        rng.shuffle(samples)
        keep = max(1, int(len(samples) * SPORTCENTER_FRACTION))
        samples = samples[:keep]

    for img_path, Hr, split in tqdm(samples, desc="sportcenter"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        pts_world = np.array([(y, -x) for x, y, _ in SMALL_COURT_POINTS], dtype=np.float32)
        types = np.array([t for _, _, t in SMALL_COURT_POINTS], dtype=int)
        pts_img = project_homography(pts_world, Hr)

        box_px = max(h, w) * BOX_SZ
        half_w = box_px / w / 2.0
        half_h = box_px / h / 2.0
        labels = []
        for (u, v), t in zip(pts_img, types):
            if box_px <= u < w - box_px and box_px <= v < h - box_px:
                nx, ny = u / w, v / h
                labels.append(f"{t} {nx:.6f} {ny:.6f} {2 * half_w:.6f} {2 * half_h:.6f}")

        if not labels:
            continue

        base = f"sport_{img_path.stem}"
        dst_img = out_dirs[f"img_{split}"] / f"{base}.jpg"
        dst_lbl = out_dirs[f"lbl_{split}"] / f"{base}.txt"
        if not dst_img.exists():
            save_jpeg(dst_img, img)
        write_label(dst_lbl, labels, "sportcenter")


def convert_deepsportradar(out_dirs):
    ds_root = Path(__file__).parent / "dataset" / "deepsportradar_instants_dataset"
    json_files = list(ds_root.glob("*/*/*.json"))
    if not json_files:
        return
    random.shuffle(json_files)
    split_idx = int(len(json_files) * 0.15)
    val_set = set(json_files[:split_idx])

    def project_points(K, R, T, pts):
        pts_cam = R @ pts.T + T.reshape(3, 1)
        uvw = K @ pts_cam
        return (uvw[:2] / uvw[2]).T

    for jpath in tqdm(json_files, desc="deepsportradar"):
        split = "val" if jpath in val_set else "train"
        meta = json.load(open(jpath))
        cal = meta["calibration"]
        K = np.array(cal["KK"], dtype=np.float32).reshape(3, 3)
        R = np.array(cal["R"], dtype=np.float32).reshape(3, 3)
        T = np.array(cal["T"], dtype=np.float32)
        base = jpath.with_suffix("")
        png_path = Path(str(base) + "_0.png")
        if not png_path.is_file():
            png_path = Path(str(base) + ".png")
        img = cv2.imread(str(png_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        pts = []
        types = []
        for x, y, t in FIBA_COURT_POINTS:
            pts.append([(x + 14.0) * 100.0, (-y + 7.5) * 100.0, 0.0])
            types.append(t)
        pts = np.array(pts, dtype=np.float32)
        types = np.array(types, dtype=int)
        pts_img = project_points(K, R, T, pts)

        box_px = max(h, w) * BOX_SZ
        half_w = box_px / w / 2.0
        half_h = box_px / h / 2.0
        labels = []
        for (u, v), t in zip(pts_img, types):
            if box_px <= u < w - box_px and box_px <= v < h - box_px:
                nx, ny = u / w, v / h
                labels.append(f"{t} {nx:.6f} {ny:.6f} {2 * half_w:.6f} {2 * half_h:.6f}")
        if not labels:
            continue

        base = f"deep_{png_path.stem}"
        dst_img = out_dirs[f"img_{split}"] / f"{base}.jpg"
        dst_lbl = out_dirs[f"lbl_{split}"] / f"{base}.txt"
        if not dst_img.exists():
            save_jpeg(dst_img, img)
        write_label(dst_lbl, labels, "deepsportradar")


def convert_roboflow(out_dirs):
    ds_root = Path(__file__).parent / "dataset" / "roboflow_court_detection"
    for split_yolo, split_out in [("train", "train"), ("valid", "val")]:
        img_dir = ds_root / split_yolo / "images"
        lbl_dir = ds_root / split_yolo / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for img_path in tqdm(images, desc=f"roboflow {split_out}"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.is_file():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(lbl_path) as f:
                line = f.readline().strip()
            if not line:
                continue
            vals = list(map(float, line.split()))
            if len(vals) < 5:
                continue
            kpt_num = (len(vals) - 5) // 3
            # cls = int(vals[0])
            kpts = np.array(vals[5 : 5 + kpt_num * 3], dtype=float).reshape(kpt_num, 3)

            labels = []
            box_px = max(h, w) * BOX_SZ
            half_w = box_px / w / 2.0
            half_h = box_px / h / 2.0
            for idx, (nx, ny, vis) in enumerate(kpts):
                if vis <= 0:
                    continue
                if idx not in MAPPING_ROBOFLOW_COURT_DETECTION:
                    continue
                t = MAPPING_ROBOFLOW_COURT_DETECTION[idx]
                labels.append(f"{t} {nx:.6f} {ny:.6f} {2 * half_w:.6f} {2 * half_h:.6f}")
            if not labels:
                continue
            base = f"rf_{img_path.stem}"
            dst_img = out_dirs[f"img_{split_out}"] / f"{base}.jpg"
            dst_lbl = out_dirs[f"lbl_{split_out}"] / f"{base}.txt"
            if not dst_img.exists():
                save_jpeg(dst_img, img)
            write_label(dst_lbl, labels, "roboflow-court-detection")


def build_data_yaml(out_root: Path):
    num_classes = int(max(p[2] for p in NBA_COURT_POINTS)) + 1
    class_names = [f"type_{i}" for i in range(num_classes)]
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root}",
                "train: images/train",
                "val: images/val",
                f"names: {class_names}",
            ]
        )
    )
    return data_yaml


def prepare_dataset(out_root=Path(__file__).parent / "yolo_combined_data", clean_output=True):
    out_root = Path(__file__).parent / "yolo_combined_data"
    if clean_output and out_root.exists():
        shutil.rmtree(out_root)
    dirs = ensure_dirs(out_root)
    convert_sportcenter(dirs)
    convert_deepsportradar(dirs)
    convert_roboflow(dirs)
    build_data_yaml(out_root)


if __name__ == "__main__":
    prepare_dataset()
