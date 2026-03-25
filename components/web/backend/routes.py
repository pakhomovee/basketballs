"""API route handlers for the basketball analytics backend."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from .database import (
    add_comment,
    create_job,
    get_comments,
    get_db,
    get_job,
    list_jobs,
)
from .models import CommentCreate, CommentResponse, JobResponse, UploadResponse
from .pipeline_runner import enqueue_job, job_dir

router = APIRouter(prefix="/api")

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB

_VIDEO_MIME: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".3gp": "video/3gpp",
    ".ts": "video/mp2t",
}


def _video_mime_type(path: Path) -> str:
    return _VIDEO_MIME.get(path.suffix.lower(), "video/mp4")


def _sanitize_filename(name: str) -> str:
    """Keep only safe characters in uploaded filename."""
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-. ]", "_", name)
    return name or "video.mp4"


# --------------------------------------------------------------------------
# Videos
# --------------------------------------------------------------------------


@router.post("/videos", response_model=UploadResponse)
async def upload_video(file: UploadFile):
    if file.filename is None:
        raise HTTPException(400, "No filename provided")

    jid = uuid.uuid4().hex[:12]
    safe_name = _sanitize_filename(file.filename)
    jdir = job_dir(jid)
    jdir.mkdir(parents=True, exist_ok=True)
    video_path = jdir / safe_name

    size = 0
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                video_path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large (max 2 GB)")
            f.write(chunk)

    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        await create_job(db, jid, safe_name, now)
    finally:
        await db.close()

    await enqueue_job(jid, str(video_path))
    return UploadResponse(job_id=jid)


@router.get("/videos")
async def list_all_videos():
    db = await get_db()
    try:
        jobs = await list_jobs(db)
    finally:
        await db.close()
    return {"jobs": [JobResponse(**j) for j in jobs]}


@router.get("/videos/{job_id}")
async def get_video_status(job_id: str):
    db = await get_db()
    try:
        job = await get_job(db, job_id)
    finally:
        await db.close()
    if not job:
        raise HTTPException(404, "Job not found")
    return JobResponse(**job)


@router.get("/videos/{job_id}/video")
async def stream_video(job_id: str, request: Request):
    """Stream the original video file with HTTP range request support."""
    db = await get_db()
    try:
        job = await get_job(db, job_id)
    finally:
        await db.close()
    if not job:
        raise HTTPException(404, "Job not found")

    jdir = job_dir(job_id)
    video_path = jdir / job["video_name"]
    if not video_path.is_file():
        raise HTTPException(404, "Video file not found")

    file_size = video_path.stat().st_size
    mime_type = _video_mime_type(video_path)
    range_header = request.headers.get("range")

    if range_header:
        # Parse Range: bytes=start-end
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if not m:
            raise HTTPException(416, "Invalid range")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            raise HTTPException(416, "Range not satisfiable")

        length = end - start + 1

        def _range_iter():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk_size = min(1024 * 1024, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            _range_iter(),
            status_code=206,
            media_type=mime_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    # Full file
    def _full_iter():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        _full_iter(),
        media_type=mime_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )


@router.get("/videos/{job_id}/annotations")
async def get_annotations(job_id: str):
    ann_path = job_dir(job_id) / "annotations.json"
    if not ann_path.is_file():
        raise HTTPException(404, "Annotations not ready or job not found")
    with open(ann_path) as f:
        data = json.load(f)
    return JSONResponse(content=data)


@router.get("/videos/{job_id}/progress")
async def get_progress(job_id: str):
    """Return current pipeline stage label and fractional progress (0–1)."""
    progress_path = job_dir(job_id) / "progress.txt"
    if progress_path.is_file():
        try:
            data = json.loads(progress_path.read_text(encoding="utf-8"))
            return {"stage": data.get("stage", ""), "pct": float(data.get("pct", 0.0))}
        except Exception:
            pass
    return {"stage": "", "pct": 0.0}


# --------------------------------------------------------------------------
# Comments
# --------------------------------------------------------------------------


@router.post("/videos/{job_id}/comments", response_model=CommentResponse)
async def create_comment(job_id: str, body: CommentCreate):
    if not body.text.strip():
        raise HTTPException(400, "Comment text cannot be empty")
    if len(body.text) > 2000:
        raise HTTPException(400, "Comment text too long (max 2000 chars)")

    db = await get_db()
    try:
        job = await get_job(db, job_id)
        if not job:
            raise HTTPException(404, "Job not found")

        now = datetime.now(timezone.utc).isoformat()
        comment_id = await add_comment(db, job_id, body.timestamp_sec, body.text.strip(), now)
    finally:
        await db.close()

    return CommentResponse(
        id=comment_id,
        job_id=job_id,
        timestamp_sec=body.timestamp_sec,
        text=body.text.strip(),
        created_at=now,
    )


@router.get("/videos/{job_id}/comments")
async def list_comments(job_id: str):
    db = await get_db()
    try:
        comments = await get_comments(db, job_id)
    finally:
        await db.close()
    return {"comments": [CommentResponse(**c) for c in comments]}
