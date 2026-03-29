"""Export pipeline annotations to a JSON-serializable dict for the web viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.classes.player import Player, PlayersDetections
from common.classes.ball import Ball, BallDetections
from common.classes.pass_event import PassEvent
from common.classes.shot_event import ShotEvent
from common.distances import cosine_dist


def _to_json_safe(value):
    """Convert numpy types to native Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 4)
    if isinstance(value, float):
        return round(value, 4)
    return value


def _serialize_player(player: Player) -> dict:
    jersey_number = None
    jersey_confidence = None
    if player.number is not None:
        jersey_number = player.number.num
        jersey_confidence = _to_json_safe(player.number.confidence)

    skeleton = None
    if player.skeleton is not None and player.skeleton.keypoints is not None:
        skeleton = np.round(player.skeleton.keypoints, 2).tolist()

    mask_polygon = None
    if player.mask_polygon is not None:
        mask_polygon = [[round(pt[0], 1), round(pt[1], 1)] for pt in player.mask_polygon]

    court_position = None
    if player.court_position is not None:
        court_position = [round(player.court_position[0], 3), round(player.court_position[1], 3)]

    return {
        "player_id": player.player_id,
        "team_id": player.team_id,
        "bbox": [int(c) for c in player.bbox] if player.bbox else None,
        "confidence": _to_json_safe(player.confidence),
        "jersey_number": jersey_number,
        "jersey_confidence": jersey_confidence,
        "court_position": court_position,
        "speed": _to_json_safe(player.speed),
        "skeleton": skeleton,
        "mask_polygon": mask_polygon,
        "is_possession": bool(getattr(player, "is_possession", False)),
    }


def _serialize_ball(ball: Ball) -> dict:
    return {
        "bbox": [int(c) for c in ball.bbox] if ball.bbox else None,
        "confidence": _to_json_safe(ball.confidence),
    }


def _compute_cross_frame_reid_matrix(
    prev_players: list[Player],
    curr_players: list[Player],
) -> dict | None:
    """Compute cross-frame REID cosine distance matrix sorted by player_id."""
    prev_with_emb = [
        p for p in prev_players if p.player_id >= 0 and (p.reid_embedding is not None or p.embedding is not None)
    ]
    curr_with_emb = [
        p for p in curr_players if p.player_id >= 0 and (p.reid_embedding is not None or p.embedding is not None)
    ]

    if not prev_with_emb or not curr_with_emb:
        return None

    prev_with_emb.sort(key=lambda p: p.player_id)
    curr_with_emb.sort(key=lambda p: p.player_id)

    row_ids = [p.player_id for p in prev_with_emb]
    col_ids = [p.player_id for p in curr_with_emb]

    distances = []
    for pa in prev_with_emb:
        emb_a = pa.reid_embedding if pa.reid_embedding is not None else pa.embedding
        row = []
        for pb in curr_with_emb:
            emb_b = pb.reid_embedding if pb.reid_embedding is not None else pb.embedding
            row.append(round(float(cosine_dist(emb_a, emb_b)), 4))
        distances.append(row)

    return {"row_ids": row_ids, "col_ids": col_ids, "distances": distances}


def _serialize_pass_event(event: PassEvent, fps: float) -> dict:
    return {
        "frame_start": event.frame_start,
        "frame_end": event.frame_end,
        "from_player_id": event.from_player_id,
        "to_player_id": event.to_player_id,
        "team_id": event.team_id,
        "timestamp_sec": round(event.frame_end / fps, 2) if fps > 0 else 0,
    }


def _serialize_shot_event(event: ShotEvent, fps: float) -> dict:
    ts_start = round(event.frame_start / fps, 2) if fps > 0 else 0.0
    ts_end = round(event.frame_end / fps, 2) if fps > 0 else 0.0
    out: dict = {
        "frame_start": event.frame_start,
        "frame_end": event.frame_end,
        "is_make": event.is_make,
        "timestamp_start_sec": ts_start,
        "timestamp_end_sec": ts_end,
    }
    if event.is_make and event.make_start is not None and event.make_end is not None:
        out["make_start"] = event.make_start
        out["make_end"] = event.make_end
        out["make_timestamp_start_sec"] = round(event.make_start / fps, 2) if fps > 0 else 0.0
        out["make_timestamp_end_sec"] = round(event.make_end / fps, 2) if fps > 0 else 0.0
    else:
        out["make_start"] = None
        out["make_end"] = None
        out["make_timestamp_start_sec"] = None
        out["make_timestamp_end_sec"] = None
    return out


def export_annotations(
    players_detections: PlayersDetections,
    ball_detections: BallDetections | None,
    video_meta: dict,
    pass_events: list[PassEvent] | None = None,
    shot_events: list[ShotEvent] | None = None,
) -> dict:
    """Build a complete annotation dict ready for JSON serialization."""
    all_frame_ids = sorted(set(players_detections.keys()) | set((ball_detections or {}).keys()))
    frames: dict[str, dict] = {}

    prev_players: list[Player] = []
    for frame_id in all_frame_ids:
        curr_players = players_detections.get(frame_id, [])
        balls = (ball_detections or {}).get(frame_id, [])

        reid_matrix = _compute_cross_frame_reid_matrix(prev_players, curr_players)

        frames[str(frame_id)] = {
            "players": [_serialize_player(p) for p in curr_players],
            "balls": [_serialize_ball(b) for b in balls],
            "reid_cross_frame_matrix": reid_matrix,
        }

        prev_players = curr_players

    fps = video_meta.get("fps", 25.0)
    serialized_passes = [_serialize_pass_event(e, fps) for e in (pass_events or [])]
    serialized_shots = [_serialize_shot_event(e, fps) for e in (shot_events or [])]

    return {
        "metadata": video_meta,
        "frames": frames,
        "pass_events": serialized_passes,
        "shot_events": serialized_shots,
    }


def save_annotations(data: dict, path: str | Path) -> None:
    """Write annotation dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
