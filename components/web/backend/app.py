"""FastAPI application entry-point for the basketball analytics backend."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import get_db, update_job_status, list_jobs
from .pipeline_runner import worker, enqueue_job, job_dir
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: recover any jobs that were interrupted mid-processing
    db = await get_db()
    try:
        jobs = await list_jobs(db)
        for job in jobs:
            if job["status"] == "processing":
                # Server restarted while this job was running — re-queue it
                video_path = job_dir(job["id"]) / job["video_name"]
                if video_path.is_file():
                    await update_job_status(db, job["id"], "queued")
                else:
                    await update_job_status(db, job["id"], "failed", error="Interrupted by server restart")
    finally:
        await db.close()

    # Start the single pipeline-worker background task
    task = asyncio.create_task(worker(get_db))

    # Re-enqueue recovered queued jobs (including those just reset above)
    db = await get_db()
    try:
        jobs = await list_jobs(db)
        for job in jobs:
            if job["status"] == "queued":
                video_path = job_dir(job["id"]) / job["video_name"]
                if video_path.is_file():
                    await enqueue_job(job["id"], str(video_path))
    finally:
        await db.close()

    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Basketball Analytics API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
