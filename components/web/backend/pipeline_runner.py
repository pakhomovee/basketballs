"""Run the basketball tracking pipeline on uploaded videos.

Imports from the existing ``components/`` package and orchestrates the same
stages as ``components/main.py``, outputting an annotation JSON file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

# Path setup — add components/ to sys.path so pipeline imports work.
COMPONENTS_DIR = Path(__file__).resolve().parent.parent.parent  # components/
WEB_DIR = Path(__file__).resolve().parent.parent  # components/web/

for _p in (str(COMPONENTS_DIR), str(WEB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"


def job_dir(job_id: str) -> Path:
    return STORAGE_DIR / "jobs" / job_id


# Progress helpers

TOTAL_STAGES = 11


def _save_progress(job_id: str, label: str, pct: float) -> None:
    try:
        (job_dir(job_id) / "progress.txt").write_text(
            json.dumps({"stage": label, "pct": round(pct, 3)}),
            encoding="utf-8",
        )
    except Exception:
        pass


@contextlib.contextmanager
def _stage(job_id: str, label: str, stage_idx: int):
    """Context manager: writes stage progress at start and end."""
    _save_progress(job_id, label, stage_idx / TOTAL_STAGES)
    try:
        yield
    finally:
        _save_progress(job_id, label, (stage_idx + 1) / TOTAL_STAGES)


def _run_pipeline(job_id: str, video_path: str) -> None:
    """Synchronous pipeline execution (runs in a thread)."""
    import cv2

    from court_detector.court_detector import CourtDetector
    from team_clustering.embedding import PlayerEmbedder
    from team_clustering.team_clustering import TeamClustering
    from tracking import FlowTracker
    from smoother import smooth_detection_coordinates
    from reidentification import extract_reid_embeddings
    from common.utils.utils import get_device
    from common.utils.models import ensure_models, get_model_paths
    from config import load_default_config
    from detector import Detector, enrich_detections_with_numbers, enrich_players_with_pose
    from detector.interpolate_ball_detections import linear_interpolate_ball_detections
    from ball_detector.detector import WASBBallDetector
    from actions.ball_possession import (
        assign_ball_possession_soft_dribble,
        apply_possession_segments,
        greedy_possession_segments_soft_dribble,
    )
    from annotation_exporter import export_annotations, save_annotations

    cfg = load_default_config()

    # Stage 0 — model check (no tqdm)
    with _stage(job_id, "Checking models…", 0):
        ensure_models(cfg)

    paths = get_model_paths(cfg)

    # Stage 1 — detect
    with _stage(job_id, "Step 1/10 — Detecting players…", 1):
        detector = Detector(model_path=str(paths.detector), conf_threshold=cfg.detector.initial_threshold)
        all_detections = detector.detect_video(video_path)

    # Stage 2 — jersey numbers
    with _stage(job_id, "Step 2/10 — Recognising jersey numbers…", 2):
        players_detections, _referees, _numbers = enrich_detections_with_numbers(
            video_path,
            all_detections,
            player_conf_threshold=cfg.detector.player_conf_threshold,
            referee_conf_threshold=cfg.detector.referee_conf_threshold,
            number_conf_threshold=cfg.detector.number_conf_threshold,
            ocr_conf_threshold=cfg.detector.ocr_conf_threshold,
        )

    # Stage 3 — pose estimation
    with _stage(job_id, "Step 3/10 — Estimating poses…", 3):
        enrich_players_with_pose(video_path, players_detections)

    # Stage 4 — court detection
    with _stage(job_id, "Step 4/10 — Detecting court…", 4):
        court_detector = CourtDetector(model_path=str(paths.court_detection), cfg=cfg)
        court_detector.run(video_path, players_detections)

    # Stage 5 — ball detection
    with _stage(job_id, "Step 5/10 — Detecting ball…", 5):
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        wasb_detector = WASBBallDetector(weights_path=str(paths.wasb), cfg=cfg)
        raw_ball_detections = wasb_detector.detect_video(video_path)
        interpolated = linear_interpolate_ball_detections(raw_ball_detections)
        ball_detections = {fid: [b] for fid, b in interpolated.items()}

    # Stage 6 — colour embeddings & mask polygons
    with _stage(job_id, "Step 6/10 — Extracting colour embeddings & masks…", 6):
        PlayerEmbedder().extract_player_embeddings(video_path, players_detections)

    # Stage 7 — ReID embeddings (optional)
    with _stage(job_id, "Step 7/10 — Extracting ReID embeddings…", 7):
        if os.path.isfile(str(paths.reid)):
            device = get_device()
            extract_reid_embeddings(video_path, players_detections, str(paths.reid), device=device)

    # Stage 8 — tracking
    with _stage(job_id, "Step 8/10 — Running tracker…", 8):
        frame_width = float(width)
        tracker = FlowTracker(cfg=cfg, frame_width=frame_width, fps=fps)
        tracker.track(players_detections)

    # Stage 9 — team clustering
    with _stage(job_id, "Step 9/10 — Clustering teams…", 9):
        team_clustering = TeamClustering()
        team_clustering.run(players_detections)

    # Stage 10 — possession, smoothing + export
    with _stage(job_id, "Step 10/10 — Possession, smoothing & exporting…", 10):
        assign_ball_possession_soft_dribble(players_detections, ball_detections)
        possession_segments = greedy_possession_segments_soft_dribble(players_detections, fps=fps)
        apply_possession_segments(players_detections, possession_segments)
        smooth_detection_coordinates(players_detections)
        video_meta = {
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "video_name": Path(video_path).name,
        }
        annotation_data = export_annotations(players_detections, ball_detections, video_meta)
        save_annotations(annotation_data, job_dir(job_id) / "annotations.json")


# Async job queue — single worker
_queue: asyncio.Queue[tuple[str, str]] | None = None


async def _get_queue() -> asyncio.Queue[tuple[str, str]]:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue()
    return _queue


async def enqueue_job(job_id: str, video_path: str) -> None:
    q = await _get_queue()
    await q.put((job_id, video_path))


async def worker(db_getter) -> None:
    """Background worker that processes one job at a time."""
    q = await _get_queue()
    while True:
        job_id, video_path = await q.get()
        db = await db_getter()
        try:
            from .database import update_job_status

            await update_job_status(db, job_id, "processing")
            await db.close()

            # Run CPU/GPU-heavy pipeline in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _run_pipeline, job_id, video_path)

            db = await db_getter()
            await update_job_status(db, job_id, "done")
        except Exception:
            err = traceback.format_exc()
            logger.error("Job %s failed:\n%s", job_id, err)
            try:
                db = await db_getter()
                await update_job_status(db, job_id, "failed", error=str(err[-500:]))
            except Exception:
                logger.error("Failed to update job status for %s", job_id)
        finally:
            try:
                await db.close()
            except Exception:
                pass
            q.task_done()
