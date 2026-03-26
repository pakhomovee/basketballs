"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class JobResponse(BaseModel):
    id: str
    status: str  # queued | processing | done | failed
    video_name: str
    display_name: str | None = None
    created_at: str
    error: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class UploadResponse(BaseModel):
    job_id: str


class CommentCreate(BaseModel):
    timestamp_sec: float
    text: str


class CommentResponse(BaseModel):
    id: int
    job_id: str
    timestamp_sec: float
    text: str
    created_at: str
