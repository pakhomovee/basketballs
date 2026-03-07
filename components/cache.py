"""Cache for detections and embeddings to avoid recomputation on repeated runs."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from common.classes.player import Player, PlayersDetections


def _video_meta(video_path: str) -> tuple[float, int]:
    """Return (mtime, size) for cache invalidation."""
    stat = os.stat(video_path)
    return (stat.st_mtime, stat.st_size)


def _detections_cache_path(video_path: str) -> Path:
    """Path for detections cache (bbox + court_position), next to the video."""
    return Path(video_path).resolve().with_suffix(".detections_cache.pkl")


def _embeddings_cache_path(video_path: str, seg_model: str) -> Path:
    """Path for embeddings cache, next to the video."""
    model_stem = Path(seg_model).stem
    return Path(video_path).resolve().with_suffix(f".embeddings_cache.{model_stem}.pkl")


def _legacy_cache_path(video_path: str, seg_model: str) -> Path:
    """Path for legacy combined cache (backward compatibility)."""
    model_stem = Path(seg_model).stem
    return Path(video_path).resolve().with_suffix(f".detections_cache.{model_stem}.pkl")


def _serialize_detections(detections: PlayersDetections, include_embeddings: bool = True) -> dict:
    """Convert PlayersDetections to a pickle-friendly structure."""
    out = {}
    for frame_id, players in detections.items():
        out[frame_id] = []
        for p in players:
            d = {"bbox": list(p.bbox), "court_position": p.court_position}
            if include_embeddings and p.embedding is not None:
                d["embedding"] = p.embedding
            out[frame_id].append(d)
    return out


def _deserialize_detections(data: dict) -> PlayersDetections:
    """Reconstruct PlayersDetections from cached structure."""
    detections: PlayersDetections = {}
    for frame_id, players_data in data.items():
        detections[frame_id] = []
        for idx, d in enumerate(players_data):
            p = Player(
                player_id=idx,
                bbox=d["bbox"],
                court_position=d.get("court_position"),
                embedding=d.get("embedding"),
            )
            detections[frame_id].append(p)
    return detections


def _serialize_embeddings(detections: PlayersDetections) -> dict:
    """Extract embeddings as {(frame_id, idx): embedding} for cache."""
    out = {}
    for frame_id, players in detections.items():
        for idx, p in enumerate(players):
            if p.embedding is not None:
                out[(frame_id, idx)] = p.embedding
    return out


def merge_embeddings(detections: PlayersDetections, embeddings: dict) -> None:
    """Merge cached embeddings into detections in place."""
    for frame_id, players in detections.items():
        for idx, p in enumerate(players):
            key = (frame_id, idx)
            if key in embeddings:
                p.embedding = embeddings[key]


def load_embeddings_cache(video_path: str, seg_model: str) -> dict | None:
    """
    Load cached embeddings if valid. Returns {(frame_id, idx): embedding} or None.
    """
    try:
        mtime, size = _video_meta(video_path)
    except OSError:
        return None

    for cache_path in [_embeddings_cache_path(video_path, seg_model), _legacy_cache_path(video_path, seg_model)]:
        if not cache_path.exists():
            continue
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
        except (pickle.PickleError, EOFError):
            continue
        meta = cached.get("meta", {})
        if meta.get("seg_model") != seg_model or meta.get("mtime") != mtime or meta.get("size") != size:
            continue
        if "embeddings" in cached:
            return cached["embeddings"]
        # Legacy: extract from detections
        detections_data = cached.get("detections", {})
        return _serialize_embeddings(_deserialize_detections(detections_data))
    return None


def load_detections_cache(
    video_path: str,
    seg_model: str,
    use_detector_cache: bool = True,
    use_embeddings_cache: bool = True,
) -> PlayersDetections | None:
    """
    Load cached detections if valid.

    When use_detector_cache: load bbox + court_position (from split or legacy cache).
    When use_embeddings_cache: load embeddings (from split or legacy cache).
    Returns None if cache is missing or invalid (e.g. video file changed).
    """
    if not use_detector_cache and not use_embeddings_cache:
        return None

    try:
        mtime, size = _video_meta(video_path)
    except OSError:
        return None

    detections: PlayersDetections | None = None
    embeddings: dict | None = None

    # Try split caches first
    if use_detector_cache:
        cache_file = _detections_cache_path(video_path)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
            except (pickle.PickleError, EOFError):
                pass
            else:
                meta = cached.get("meta", {})
                if meta.get("mtime") == mtime and meta.get("size") == size:
                    detections = _deserialize_detections(cached["detections"])

    if use_embeddings_cache and (detections is not None or use_detector_cache):
        embeddings = load_embeddings_cache(video_path, seg_model)

    # If no detections from split cache, try legacy combined cache
    if detections is None and use_detector_cache:
        cache_file = _legacy_cache_path(video_path, seg_model)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
            except (pickle.PickleError, EOFError):
                pass
            else:
                meta = cached.get("meta", {})
                if meta.get("seg_model") == seg_model and meta.get("mtime") == mtime and meta.get("size") == size:
                    detections = _deserialize_detections(cached["detections"])
                    if use_embeddings_cache and embeddings is None and detections:
                        embeddings = _serialize_embeddings(detections)

    if detections is None:
        return None

    if embeddings and use_embeddings_cache:
        merge_embeddings(detections, embeddings)
    elif not use_embeddings_cache:
        for players in detections.values():
            for p in players:
                p.embedding = None

    return detections


def save_detections_cache(
    video_path: str,
    detections: PlayersDetections,
    seg_model: str,
    use_detector_cache: bool = True,
    use_embeddings_cache: bool = True,
) -> None:
    """Save detections to cache (split or combined based on flags)."""
    try:
        mtime, size = _video_meta(video_path)
    except OSError:
        return

    meta = {
        "video_path": str(Path(video_path).resolve()),
        "mtime": mtime,
        "size": size,
    }

    if use_detector_cache:
        cache_file = _detections_cache_path(video_path)
        # Save without embeddings for split format
        det_serialized = _serialize_detections(detections, include_embeddings=False)
        cached = {"meta": meta, "detections": det_serialized}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

    if use_embeddings_cache:
        cache_file = _embeddings_cache_path(video_path, seg_model)
        cached = {
            "meta": {**meta, "seg_model": seg_model},
            "embeddings": _serialize_embeddings(detections),
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
