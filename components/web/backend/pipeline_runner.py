"""Run the basketball tracking pipeline on uploaded videos.

Uses the shared pipeline implementation in ``components/run_pipeline.py`` and
exports annotations for the web viewer.
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

from run_pipeline import TOTAL_STAGES as PIPELINE_TOTAL_STAGES, run_pipeline

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


def _save_progress(job_id: str, label: str, pct: float) -> None:
    try:
        (job_dir(job_id) / "progress.txt").write_text(
            json.dumps({"stage": label, "pct": round(pct, 3)}),
            encoding="utf-8",
        )
    except Exception:
        pass


class _JobStageLogger:
    def __init__(self, job_id: str, total_stages: int):
        self.job_id = job_id
        self.total_stages = total_stages

    def set_stage(self, label: str, stage_idx: int) -> None:
        pct = float(stage_idx) / float(self.total_stages) if self.total_stages > 0 else 0.0
        _save_progress(self.job_id, label, pct)


def _run_pipeline(job_id: str, video_path: str) -> None:
    """Synchronous pipeline execution (runs in a thread)."""
    from config import load_default_config
    from annotation_exporter import export_annotations, save_annotations

    cfg = load_default_config()

    total_stages = PIPELINE_TOTAL_STAGES + 1  # pipeline + export
    stage_logger = _JobStageLogger(job_id, total_stages=total_stages)
    result = run_pipeline(video_path, cfg, stage_logger=stage_logger)

    # Export is an extra step on top of the shared pipeline.
    stage_logger.set_stage("Exporting annotations…", PIPELINE_TOTAL_STAGES)
    annotation_data = export_annotations(result.players_detections, result.ball_detections, result.video_meta)
    save_annotations(annotation_data, job_dir(job_id) / "annotations.json")
    stage_logger.set_stage("Done", total_stages)


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
