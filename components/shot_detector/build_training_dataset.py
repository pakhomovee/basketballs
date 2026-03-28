"""
Build per-clip training features for shot_detector.

For each annotated clip we compute:
1) homographies (frame -> court) using ``CourtDetector.extract_homographies_from_video_v2``
2) rim detections using ``components/detector`` + ``get_video_rim_detections``
3) ball detections using ``WASBBallDetector.detect_video``

Output format:
- one ``.npz`` sample per clip with fixed-size arrays:
  homographies, rim/ball bboxes + confidences, and court coordinates.
- an ``index.jsonl`` describing each sample and labels (shot/make segments).

The script is robust: errors in individual clips are caught, logged, and the
pipeline continues.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil

import cv2
import numpy as np

from config import AppConfig, load_default_config
from court_detector.court_constants import CourtConstants
from court_detector.court_detector import CourtDetector, project_homography
from common.classes import CourtType
from common.classes.ball import Ball
from common.utils.models import ensure_models, get_model_paths
from detector import Detector, get_video_rim_detections
from ball_detector.detector import WASBBallDetector


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class ClipSample:
    clip_path: str
    sample_path: str
    status: str
    start_frame: int
    finish_frame: int
    shot_start_frame: int | None
    shot_end_frame: int | None
    make_start_frame: int | None
    make_end_frame: int | None


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "errors.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def _read_annotations(annotations_path: Path) -> list[dict]:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    return list(data.get("annotations", []))


def _safe_int(v, default: int) -> int:
    if v is None:
        return default
    return int(v)


def _clip_range_from_annotation(rec: dict, total_frames: int) -> tuple[int, int]:
    start_frame = rec.get("start_frame")
    finish_frame = rec.get("finish_frame")

    if start_frame is None:
        start_frame = 0
    if finish_frame is None:
        finish_frame = max(total_frames - 1, 0)

    start_frame = int(start_frame)
    finish_frame = int(finish_frame)

    start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
    finish_frame = max(0, min(finish_frame, max(total_frames - 1, 0)))
    if finish_frame < start_frame:
        start_frame = 0
        finish_frame = max(total_frames - 1, 0)
    return start_frame, finish_frame


def _get_video_meta(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, w, h, total


def _crop_video_by_frames(
    video_path: Path,
    start_frame: int,
    finish_frame: int,
    out_path: Path,
) -> tuple[float, int, int, int]:
    """
    Crop [start_frame, finish_frame] (inclusive) into out_path.

    Returns
    -------
    (fps, width, height, written_frames)
    """
    if finish_frame < start_frame:
        raise ValueError(f"Invalid crop range: {start_frame}..{finish_frame}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Cannot read start frame {start_frame} from {video_path}")

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {out_path}")

    written = 0
    # write first frame
    writer.write(frame)
    written += 1

    # write remaining frames
    for _ in range(start_frame + 1, finish_frame + 1):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    return fps, w, h, written


def _select_best_rim(rims: list, frame_id: int) -> tuple[list[float], float] | None:
    if not rims:
        return None
    best = max(rims, key=lambda d: getattr(d, "confidence", 0.0))
    bbox = best.get_bbox()  # [x1,y1,x2,y2]
    conf = float(getattr(best, "confidence", 0.0))
    return bbox, conf


def _ball_center_bottom(ball: Ball) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in ball.bbox[:4]]
    cx = 0.5 * (x1 + x2)
    cy = y2  # bottom
    return cx, cy


def build_training_dataset(
    dataset_dir: Path,
    annotations_path: Path,
    out_dir: Path,
    *,
    cfg: AppConfig | None = None,
    include_invalid: bool = False,
    court_type: str = "nba",
    rim_conf_threshold: float = 0.1,
    ball_step: int = 3,
    slice_to_clip_range: bool = True,
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    if cfg is None:
        cfg = load_default_config()

    _setup_logging(out_dir)
    log = logging.getLogger(__name__)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    annotations = _read_annotations(annotations_path)
    if limit is not None:
        annotations = annotations[:limit]

    # Models: ensure and create once (reused for all clips)
    ensure_models(cfg)
    paths = get_model_paths(cfg)

    court_type_norm = court_type.lower()
    if court_type_norm == "nba":
        ct = CourtType.NBA
    elif court_type_norm == "fiba":
        ct = CourtType.FIBA
    else:
        raise ValueError(f"Unsupported court_type: {court_type}")

    court_constants = CourtConstants(ct)
    court_detector = CourtDetector(cfg=cfg)
    detector = Detector(model_path=str(paths.detector), conf_threshold=rim_conf_threshold)
    wasb_detector = WASBBallDetector(cfg=cfg, step=ball_step)

    index_path = out_dir / "index.jsonl"
    index_path.write_text("", encoding="utf-8")

    created = 0
    skipped = 0

    tmp_root = Path(tempfile.mkdtemp(prefix="shot_detector_crops_"))
    try:
        for rec in annotations:
            clip_path = str(rec.get("clip_path", ""))
            if not clip_path:
                continue
            status = rec.get("status", "ok")
            if status == "invalid" and not include_invalid:
                continue

            video_path = dataset_dir / clip_path
            if not video_path.exists():
                log.warning("Clip not found, skipping: %s", video_path)
                continue

            sample_id = Path(clip_path).stem.replace(" ", "_")
            sample_path = samples_dir / f"{sample_id}.npz"
            if sample_path.exists() and not overwrite:
                log.info("Sample exists, skipping: %s", sample_path.name)
                skipped += 1
                continue

            try:
                _, frame_w, frame_h, total_frames_cap = _get_video_meta(video_path)
                start_frame, finish_frame = _clip_range_from_annotation(rec, total_frames_cap)

                crop_path = tmp_root / f"{sample_id}_{start_frame}_{finish_frame}.mp4"
                fps_crop, crop_w, crop_h, written = _crop_video_by_frames(video_path, start_frame, finish_frame, crop_path)
                if written <= 0:
                    raise RuntimeError("Crop video has no frames")

                # Homographies on the cropped video (local frame indices 0..T-1)
                homographies, frames_sizes, _, _ = court_detector.extract_homographies_from_video_v2(
                    str(crop_path),
                    court_constants,
                    smoothness_cost=cfg.court_detector.smoothness_cost,
                    keypoint_eps=cfg.court_detector.keypoint_eps,
                    smoothing_num_epochs=cfg.court_detector.smoothing_num_epochs,
                    smoothing_lr=cfg.court_detector.smoothing_lr,
                )

                n_h = len(homographies)
                n_fs = len(frames_sizes)
                if n_h <= 0 or n_fs <= 0:
                    raise RuntimeError("No frames for homography extraction")

                T = min(n_h, n_fs)
                homographies = homographies[:T]
                frames_sizes = frames_sizes[:T]

                # Rim detections on the cropped video
                video_detections = detector.detect_video(str(crop_path))
                rims_by_frame = get_video_rim_detections(video_detections, conf_threshold=rim_conf_threshold)

                # Ball detections on the cropped video (already interpolated)
                ball_detections = wasb_detector.detect_video(str(crop_path))  # local frame_id -> Ball

                # Allocate fixed arrays
                homography_arr = np.zeros((T, 3, 3), dtype=np.float32)
                homography_mask = np.zeros((T,), dtype=np.float32)
                rim_bbox = np.zeros((T, 4), dtype=np.float32)
                rim_conf = np.zeros((T,), dtype=np.float32)
                ball_bbox = np.zeros((T, 4), dtype=np.float32)
                ball_conf = np.zeros((T,), dtype=np.float32)

                rim_court_xy = np.full((T, 2), np.nan, dtype=np.float32)
                ball_court_xy = np.full((T, 2), np.nan, dtype=np.float32)

                court_w, court_h = court_constants.court_size

                for local_f in range(T):
                    H = homographies[local_f]
                    w_f, h_f = frames_sizes[local_f]
                    w_f = max(int(w_f), 1)
                    h_f = max(int(h_f), 1)

                    if H is not None:
                        homography_arr[local_f] = H.astype(np.float32)
                        homography_mask[local_f] = 1.0

                    # Rim (top-1 by confidence)
                    rims = rims_by_frame.get(local_f, [])
                    best_rim = _select_best_rim(rims, local_f)
                    if best_rim is not None:
                        rbbox, rconf = best_rim
                        rim_bbox[local_f] = np.array(rbbox, dtype=np.float32)
                        rim_conf[local_f] = rconf

                        if H is not None:
                            x1, y1, x2, y2 = [float(v) for v in rbbox[:4]]
                            cx = 0.5 * (x1 + x2)
                            cy = y2
                            x_norm = cx / w_f
                            y_norm = cy / h_f
                            pt = project_homography(np.array([[x_norm, y_norm]], dtype=np.float32), H)[0]
                            rim_court_xy[local_f] = np.array([pt[0] * court_w, pt[1] * court_h], dtype=np.float32)

                    # Ball
                    ball = ball_detections.get(local_f)
                    if ball is not None:
                        bbbox = ball.bbox[:4]
                        ball_bbox[local_f] = np.array(bbbox, dtype=np.float32)
                        ball_conf[local_f] = float(ball.confidence or 0.0)

                        if H is not None:
                            cx, cy = _ball_center_bottom(ball)
                            x_norm = cx / w_f
                            y_norm = cy / h_f
                            pt = project_homography(np.array([[x_norm, y_norm]], dtype=np.float32), H)[0]
                            ball_court_xy[local_f] = np.array([pt[0] * court_w, pt[1] * court_h], dtype=np.float32)

                # Labels relative to crop
                def _rel(label_name: str) -> int | None:
                    v = rec.get(label_name)
                    if v is None:
                        return None
                    v = int(v) - start_frame
                    if v < 0 or v >= T:
                        return None
                    return v

                clip_sample = ClipSample(
                    clip_path=clip_path,
                    sample_path=str(sample_path.relative_to(out_dir)),
                    status=status,
                    start_frame=start_frame,
                    finish_frame=finish_frame,
                    shot_start_frame=_rel("shot_start_frame"),
                    shot_end_frame=_rel("shot_end_frame"),
                    make_start_frame=_rel("make_start_frame"),
                    make_end_frame=_rel("make_end_frame"),
                )

                np.savez_compressed(
                    sample_path,
                    fps=float(fps_crop),
                    frame_count=int(T),
                    frame_offset=int(start_frame),
                    homography=homography_arr,
                    homography_mask=homography_mask,
                    rim_bbox=rim_bbox,
                    rim_conf=rim_conf,
                    rim_court_xy=rim_court_xy,
                    ball_bbox=ball_bbox,
                    ball_conf=ball_conf,
                    ball_court_xy=ball_court_xy,
                    status=str(status),
                    shot_start_frame=-1 if clip_sample.shot_start_frame is None else int(clip_sample.shot_start_frame),
                    shot_end_frame=-1 if clip_sample.shot_end_frame is None else int(clip_sample.shot_end_frame),
                    make_start_frame=-1 if clip_sample.make_start_frame is None else int(clip_sample.make_start_frame),
                    make_end_frame=-1 if clip_sample.make_end_frame is None else int(clip_sample.make_end_frame),
                )

                with index_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {**clip_sample.__dict__, "sample_path_abs": str(sample_path)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                created += 1
                log.info("Built sample: %s (%d frames)", sample_path.name, T)

            except Exception as e:
                err = traceback.format_exc()
                log.error("Failed for clip %s: %s\n%s", clip_path, e, err)
                continue

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    log.info("Done. created=%d skipped=%d out_dir=%s", created, skipped, out_dir)


def _parse_args():
    import argparse

    default_dataset_dir = Path(__file__).resolve().parent / "dataset"
    default_annotations = default_dataset_dir / "annotations.json"
    default_out = Path(__file__).resolve().parent / "dataset_features"

    p = argparse.ArgumentParser(description="Build shot_detector training dataset")
    p.add_argument("--dataset-dir", type=Path, default=default_dataset_dir, help="Path to shot_detector/dataset")
    p.add_argument("--annotations", type=Path, default=default_annotations, help="Path to annotations.json")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Output directory")
    p.add_argument("--config", type=Path, default=None, help="Optional AppConfig YAML")
    p.add_argument("--include-invalid", action="store_true", help="Include status=invalid clips")
    p.add_argument("--court-type", type=str, default="nba", choices=["nba", "fiba"], help="Court type")
    p.add_argument("--rim-conf", type=float, default=0.1, help="Detector conf threshold for rims")
    p.add_argument("--ball-step", type=int, default=3, help="Triplet stride for WASB ball detector")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing samples")
    p.add_argument("--limit", type=int, default=None, help="Limit number of clips for debugging")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_default_config() if args.config is None else load_default_config()  # keep defaults if custom path is added later
    if args.config is not None:
        # Late import to avoid circular imports for older environments
        from config import load_app_config

        cfg = load_app_config(args.config)

    build_training_dataset(
        dataset_dir=args.dataset_dir,
        annotations_path=args.annotations,
        out_dir=args.out_dir,
        cfg=cfg,
        include_invalid=args.include_invalid,
        court_type=args.court_type,
        rim_conf_threshold=args.rim_conf,
        ball_step=args.ball_step,
        overwrite=args.overwrite,
        limit=args.limit,
    )

