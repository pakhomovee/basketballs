from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm

from common.classes.detections import VideoDetections
from common.classes.number import NumberDetections
from common.classes.player import PlayersDetections, Player
from common.classes.referee import RefereesDetections

from common.classes.skeleton import Skeleton
from common.distances import bbox_iou

from detector.detector import (
    get_frame_number_detections,
    get_frame_pose_detections,
    get_video_players_detections,
    get_video_referee_detections,
    PoseDetection,
)


def _bbox_inside(inner: list[int], outer: list[int]) -> bool:
    """True if inner bbox [x1,y1,x2,y2] is entirely inside outer bbox."""
    if len(inner) < 4 or len(outer) < 4:
        return False
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def match_numbers_to_players(
    players_detections: PlayersDetections,
    number_detections: NumberDetections,
    referees_detections: RefereesDetections,
) -> None:
    """
    For each number, assign it to a player only if exactly one player bbox contains it.
    If the number lies inside several players, it is not assigned to anyone.
    If the number is inside any referee bbox, it is not assigned to any player.
    Modifies players in place.
    """
    for frame_id, numbers in number_detections.items():
        players = players_detections.get(frame_id, [])
        referees = referees_detections.get(frame_id, [])
        for player in players:
            player.number = None
        for number in numbers:
            if any(_bbox_inside(number.bbox, r.bbox) for r in referees):
                continue
            containing: list[Player] = []
            for player in players:
                if _bbox_inside(number.bbox, player.bbox):
                    containing.append(player)
            if not containing:
                continue
            if len(containing) > 1:
                continue
            p = containing[0]
            if p.number is not None:
                continue
            p.number = number


def match_poses_to_players(
    players: list[Player],
    poses: list[PoseDetection],
    *,
    iou_threshold: float = 0.7,
) -> None:
    """
    Match pose detections to players by IoU in descending order (greedy one-to-one).

    Computes IoU for every (player, pose) pair, sorts all pairs by IoU descending,
    then greedily assigns matches while enforcing:
      - one pose can match at most one player
      - one player can match at most one pose
    Only pairs with IoU >= iou_threshold are considered.
    """
    for p in players:
        p.skeleton = None

    candidates: list[tuple[float, int, int]] = []  # (iou, player_idx, pose_idx)
    for p_idx, player in enumerate(players):
        if not player.bbox:
            continue
        for pose_idx, pose in enumerate(poses):
            iou = bbox_iou(player.bbox, pose.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, pose_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_players: set[int] = set()
    matched_poses: set[int] = set()
    for _, p_idx, pose_idx in candidates:
        if p_idx in matched_players or pose_idx in matched_poses:
            continue
        players[p_idx].skeleton = Skeleton(keypoints=poses[pose_idx].keypoints)
        matched_players.add(p_idx)
        matched_poses.add(pose_idx)


def enrich_detections_with_numbers(
    video: cv2.VideoCapture,
    video_detections: VideoDetections,
    *,
    player_conf_threshold: float = 0.1,
    referee_conf_threshold: float = 0.25,
    number_conf_threshold: float = 0.25,
    ocr_conf_threshold: float = 0.999,
) -> tuple[PlayersDetections, RefereesDetections, NumberDetections]:
    """
    Run number recognition on each frame and assign numbers to players.
    Reads video frames, runs OCR on number bboxes via recognize_numbers_in_frame (inside
    get_frame_number_detections), then match_numbers_to_players. Mutates and returns
    players_detections with .number set where a number was assigned.
    """
    players_detections = get_video_players_detections(video_detections, conf_threshold=player_conf_threshold)
    referees_detections = get_video_referee_detections(video_detections, conf_threshold=referee_conf_threshold)
    number_detections: NumberDetections = {}
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    for i in tqdm(range(frame_count), desc="Enrich numbers"):
        ret, frame = video.read()
        if not ret:
            break
        if idx == len(video_detections):
            break
        if video_detections[idx].frame_id != i:
            continue

        number_detections[i] = get_frame_number_detections(
            video_detections[idx],
            frame=frame,
            conf_threshold=number_conf_threshold,
            ocr_conf_threshold=ocr_conf_threshold,
        )
        idx += 1
    match_numbers_to_players(players_detections, number_detections, referees_detections)
    return players_detections, referees_detections, number_detections


def enrich_players_with_pose(
    video: cv2.VideoCapture,
    players_detections: PlayersDetections,
    *,
    pose_model_path: str | Path | None = None,
    pose_conf_threshold: float = 0.15,
    match_iou_threshold: float = 0.3,
) -> PlayersDetections:
    """
    Read video frames and enrich provided players_detections with pose skeletons.
    Mutates Player objects in-place and also returns players_detections.
    """
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in tqdm(range(frame_count), desc="Pose enrich", unit="frame"):
        ret, frame = video.read()
        if not ret:
            break
        players_in_frame = players_detections.get(frame_id, [])
        if not players_in_frame:
            continue
        poses = get_frame_pose_detections(frame, model_path=pose_model_path, conf_threshold=pose_conf_threshold)
        match_poses_to_players(players_in_frame, poses, iou_threshold=match_iou_threshold)
    return players_detections


# Backwards/wording-friendly alias (optional)
enrich_players_with_numbers = enrich_detections_with_numbers
