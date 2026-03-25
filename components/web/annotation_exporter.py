"""Export pipeline annotations to a JSON-serializable dict for the web viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.classes.player import Player, PlayersDetections
from common.classes.ball import Ball, BallDetections
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


def export_annotations(
    players_detections: PlayersDetections,
    ball_detections: BallDetections | None,
    video_meta: dict,
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

    return {"metadata": video_meta, "frames": frames}


def save_annotations(data: dict, path: str | Path) -> None:
    """Write annotation dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
