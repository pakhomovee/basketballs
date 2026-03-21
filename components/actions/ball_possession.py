from __future__ import annotations

from collections import Counter

from common.classes.ball import BallDetections
from common.classes.player import Player, PlayersDetections


def _get_hand_points(
    player: Player,
    *,
    left_wrist_idx: int = 9,
    right_wrist_idx: int = 10,
    conf_threshold: float = 0.1,
) -> list[tuple[float, float]]:
    """Return valid wrist points from player skeleton as (x, y)."""
    if player.skeleton is None or player.skeleton.keypoints is None:
        return []
    keypoints = player.skeleton.keypoints
    if keypoints.size == 0:
        return []

    num_kp = keypoints.shape[0]
    points: list[tuple[float, float]] = []
    for idx in (left_wrist_idx, right_wrist_idx):
        if idx >= num_kp:
            continue
        x, y, c = keypoints[idx]
        if c >= conf_threshold:
            points.append((float(x), float(y)))
    return points


def assign_ball_possession(
    players_detections: PlayersDetections,
    ball_detections: BallDetections,
    *,
    max_hand_ball_dist: float = 100.0,
    wrist_conf_threshold: float = 0.1,
) -> PlayersDetections:
    """
    Compute raw (soft) ball possession confidence for each frame.

    For each frame:
      - reset all players `has_ball_raw=False` and `possession_conf_raw=0.0`
      - pick the most confident ball detection in that frame
      - for each player, use nearest wrist distance to ball center and convert to confidence:
            conf = max(0, 1 - dist / max_hand_ball_dist)
      - store confidence in `player.possession_conf_raw`

    This stage does not force a unique owner; multiple players may have positive confidence.
    For compatibility, `has_ball_raw` is set True when `possession_conf_raw > 0`.

    Mutates `players_detections` in place and also returns it.
    """
    max_dist_sq = max_hand_ball_dist * max_hand_ball_dist

    for frame_id, players in players_detections.items():
        for player in players:
            player.has_ball_raw = False
            player.possession_conf_raw = 0.0

        balls_in_frame = ball_detections.get(frame_id, [])
        if not balls_in_frame:
            continue

        best_ball = max(balls_in_frame, key=lambda b: b.confidence or 0.0)
        if len(best_ball.bbox) < 4:
            continue
        bx1, by1, bx2, by2 = best_ball.bbox[:4]
        ball_cx = (bx1 + bx2) / 2.0
        ball_cy = (by1 + by2) / 2.0

        for player in players:
            hand_points = _get_hand_points(player, conf_threshold=wrist_conf_threshold)
            if not hand_points:
                continue
            player_best_sq = float("inf")
            for hx, hy in hand_points:
                dist_sq = (hx - ball_cx) ** 2 + (hy - ball_cy) ** 2
                if dist_sq < player_best_sq:
                    player_best_sq = dist_sq
            if player_best_sq <= max_dist_sq:
                dist = player_best_sq**0.5
                conf = max(0.0, 1.0 - dist / max_hand_ball_dist)
                player.possession_conf_raw = conf
                player.has_ball_raw = conf > 0.0

    # Keep legacy combined flag for backward compatibility.
    for players in players_detections.values():
        for p in players:
            p.has_ball = p.has_ball_raw or p.has_ball_post

    return players_detections


def _frame_conf_map(players: list[Player]) -> dict[int, float]:
    """Return per-frame possession confidence map: player_id -> confidence (>0 only)."""
    conf = {p.player_id: float(p.possession_conf_raw) for p in players if p.possession_conf_raw > 0.0}
    return conf


def _segment_ok(
    frame_confs: list[dict[int, float]],
    min_length: int,
    other_max_share: float,
    window_size: int = 30,
) -> tuple[bool, int | None, Counter]:
    """
    Validate candidate segment by confidence-share rule.

    Rule:
      - segment length >= min_length
      - there exists one dominant player D by total confidence
      - any other player has total confidence in no more than other_max_share * segment_length
      - segment starts and ends on frames where D has positive confidence
      - for every consecutive `window_size` frames, D has strictly more
        total confidence than any other player in that window.
    """
    n = len(frame_confs)
    if n < min_length:
        return False, None, Counter()

    # Aggregate confidence over the whole segment.
    total_conf: Counter = Counter()
    for c in frame_confs:
        for pid, v in c.items():
            total_conf[pid] += float(v)
    if not total_conf:
        return False, None, Counter()

    dominant, _ = total_conf.most_common(1)[0]

    # Segment must start and end on frames with dominant positive confidence.
    if frame_confs[0].get(dominant, 0.0) <= 0.0 or frame_confs[-1].get(dominant, 0.0) <= 0.0:
        return False, None, total_conf

    max_other = other_max_share * n
    for pid, c in total_conf.items():
        if pid == dominant:
            continue
        if c > max_other:
            return False, None, total_conf

    # In every sliding window, dominant owner must have strictly more total confidence.
    if window_size > 0 and n >= window_size:
        for start in range(0, n - window_size + 1):
            win = frame_confs[start : start + window_size]
            win_conf: Counter = Counter()
            for c in win:
                for pid, v in c.items():
                    win_conf[pid] += float(v)
            dom_win = win_conf.get(dominant, 0.0)
            max_other_win = max((c for pid, c in win_conf.items() if pid != dominant), default=0.0)
            if dom_win <= max_other_win:
                return False, None, total_conf

    return True, dominant, total_conf


def greedy_possession_segments(
    players_detections: PlayersDetections,
    *,
    min_length: int = 30,
    other_max_share: float = 0.2,
    window_size: int = 30,
) -> list[dict]:
    """
    Greedily extract possession segments from start to end of video.

    Uses `player.possession_conf_raw` in each frame (set by `assign_ball_possession`).
    At each start position, picks the longest valid segment [start, end] such that:
      - segment length >= min_length
      - all players except one have total confidence <= other_max_share * segment_length.
      - segment starts and ends with owner positive confidence.
      - in every sliding window of `window_size` frames, owner has strictly more
        total confidence than any other player.

    If no valid segment starts at current frame, advances start by 1 frame.

    Returns:
        list of dicts:
          {
            "start_frame": int,
            "end_frame": int,
            "owner_player_id": int,
            "length": int,
            "owner_share": float,
          }
    """
    frame_ids = sorted(players_detections.keys())
    if not frame_ids:
        return []

    frame_confs = [_frame_conf_map(players_detections[fid]) for fid in frame_ids]
    n = len(frame_ids)
    segments: list[dict] = []

    i = 0
    while i < n:
        best_j = -1
        best_owner: int | None = None
        best_cnt = Counter()

        # longest available segment starting at i
        for j in range(n - 1, i + min_length - 2, -1):
            ok, owner, cnt = _segment_ok(
                frame_confs[i : j + 1],
                min_length=min_length,
                other_max_share=other_max_share,
                window_size=window_size,
            )
            if ok and owner is not None:
                best_j = j
                best_owner = owner
                best_cnt = cnt
                break

        if best_j == -1 or best_owner is None:
            i += 1
            continue

        length = best_j - i + 1
        owner_frames = best_cnt.get(best_owner, 0.0)
        segments.append(
            {
                "start_frame": frame_ids[i],
                "end_frame": frame_ids[best_j],
                "owner_player_id": best_owner,
                "length": length,
                "owner_share": owner_frames / float(length),
            }
        )
        i = best_j + 1

    return segments


def apply_possession_segments(
    players_detections: PlayersDetections,
    segments: list[dict],
) -> PlayersDetections:
    """
    Rewrite per-frame postprocessed possession flags using extracted possession segments.

    Behavior:
      - reset `has_ball_post=False` and `possession_conf_post=0.0` for all players
      - for each segment, set `has_ball_post=True` and `possession_conf_post=1.0`
        only for `owner_player_id`
        in frames [start_frame, end_frame]

    Mutates `players_detections` in place and also returns it.
    """
    # Reset postprocessed flags first.
    for players in players_detections.values():
        for p in players:
            p.has_ball_post = False
            p.possession_conf_post = 0.0

    if not segments:
        return players_detections

    # Fast frame access by frame id.
    for seg in segments:
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        owner_id = int(seg["owner_player_id"])
        if end < start:
            continue
        for frame_id in range(start, end + 1):
            players = players_detections.get(frame_id, [])
            for p in players:
                p.has_ball_post = p.player_id == owner_id
                p.possession_conf_post = 1.0 if p.player_id == owner_id else 0.0

    # Keep legacy combined flag for backward compatibility.
    for players in players_detections.values():
        for p in players:
            p.has_ball = p.has_ball_raw or p.has_ball_post

    return players_detections


def assign_ball_possession_soft_dribble(
    players_detections: PlayersDetections,
    ball_detections: BallDetections,
    *,
    bbox_expand_ratio: float = 0.05,
    min_expand_px: int = 3,
) -> PlayersDetections:
    """
    Alternative raw possession based only on `is_dribble`.

    This variant intentionally ignores skeletons and hand distance:
      - if player.is_dribble is True -> has_ball_raw = True
      - else -> has_ball_raw = False

    Additional override rule:
      - if ball center is inside expanded bbox of exactly one player in frame,
        this player becomes the only raw owner for that frame.
    """
    for frame_id, players in players_detections.items():
        for player in players:
            player.has_ball_raw = bool(getattr(player, "is_dribble", False))
            # Keep legacy field populated, but binary (no confidence weighting).
            player.possession_conf_raw = 1.0 if player.has_ball_raw else 0.0

        balls_in_frame = ball_detections.get(frame_id, [])
        if not balls_in_frame:
            continue
        print(balls_in_frame)
        best_ball = max(balls_in_frame, key=lambda b: b.confidence or 0.0)
        if len(best_ball.bbox) < 4:
            continue
        bx1, by1, bx2, by2 = best_ball.bbox[:4]
        ball_cx = (bx1 + bx2) / 2.0
        ball_cy = (by1 + by2) / 2.0

        owners: list[Player] = []
        for player in players:
            if len(player.bbox) < 4:
                continue
            x1, y1, x2, y2 = player.bbox[:4]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            expand = max(min_expand_px, int(max(w, h) * bbox_expand_ratio))
            ex1 = x1 - expand
            ey1 = y1 - expand
            ex2 = x2 + expand
            ey2 = y2 + expand
            if ex1 <= ball_cx <= ex2 and ey1 <= ball_cy <= ey2:
                owners.append(player)

        # If ball is inside exactly one expanded player bbox, make it unique owner.
        if len(owners) == 1:
            owner_id = owners[0].player_id
            for player in players:
                player.has_ball_raw = player.player_id == owner_id
                player.possession_conf_raw = 1.0 if player.has_ball_raw else 0.0

    # Keep legacy combined flag for backward compatibility.
    for players in players_detections.values():
        for p in players:
            p.has_ball = p.has_ball_raw or p.has_ball_post

    return players_detections


def _segment_ok_soft(
    frame_hits: list[dict[int, int]],
    min_length: int,
    other_max_share: float,
    window_size: int = 40,
    min_owner_share: float = 0.55,
) -> tuple[bool, int | None, Counter]:
    """
    Softer segment validation than `_segment_ok`.

    Softened segment validation for dribble-based binary hits.

    Conditions:
      - dominant owner share >= min_owner_share across full segment
      - segment starts/ends on frames where dominant is present
      - other players are bounded by other_max_share
      - in each sliding window dominant is strictly above every other player
    """
    n = len(frame_hits)
    if n < min_length:
        return False, None, Counter()

    total_hits: Counter = Counter()
    for c in frame_hits:
        for pid, v in c.items():
            total_hits[pid] += int(v)
    if not total_hits:
        return False, None, Counter()

    dominant, dominant_total = total_hits.most_common(1)[0]
    seg_total = sum(total_hits.values())
    if seg_total <= 0.0:
        return False, None, total_hits
    if (dominant_total / seg_total) < min_owner_share:
        return False, None, total_hits

    if frame_hits[0].get(dominant, 0) <= 0 or frame_hits[-1].get(dominant, 0) <= 0:
        return False, None, total_hits

    max_other = other_max_share * n
    for pid, c in total_hits.items():
        if pid == dominant:
            continue
        if c > max_other:
            return False, None, total_hits

    if window_size > 0 and n >= window_size:
        for start in range(0, n - window_size + 1):
            win = frame_hits[start : start + window_size]
            win_hits: Counter = Counter()
            for c in win:
                for pid, v in c.items():
                    win_hits[pid] += int(v)
            dom_win = win_hits.get(dominant, 0)
            max_other_win = max((c for pid, c in win_hits.items() if pid != dominant), default=0)
            if dom_win <= max_other_win:
                return False, None, total_hits

    return True, dominant, total_hits


def greedy_possession_segments_soft_dribble(
    players_detections: PlayersDetections,
    *,
    min_length: int = 40,
    other_max_share: float = 0.35,
    window_size: int = 40,
    min_owner_share: float = 0.55,
) -> list[dict]:
    """
    Greedy segment extraction over dribble-only binary raw possession.

    Designed for output of `assign_ball_possession_soft_dribble`.
    """
    frame_ids = sorted(players_detections.keys())
    if not frame_ids:
        return []

    frame_hits = [
        {p.player_id: 1 for p in players_detections[fid] if getattr(p, "has_ball_raw", False)} for fid in frame_ids
    ]
    n = len(frame_ids)
    segments: list[dict] = []

    i = 0
    while i < n:
        best_j = -1
        best_owner: int | None = None
        best_cnt = Counter()

        for j in range(n - 1, i + min_length - 2, -1):
            ok, owner, cnt = _segment_ok_soft(
                frame_hits[i : j + 1],
                min_length=min_length,
                other_max_share=other_max_share,
                window_size=window_size,
                min_owner_share=min_owner_share,
            )
            if ok and owner is not None:
                best_j = j
                best_owner = owner
                best_cnt = cnt
                break

        if best_j == -1 or best_owner is None:
            i += 1
            continue

        length = best_j - i + 1
        owner_hits = best_cnt.get(best_owner, 0)
        total_hits = sum(best_cnt.values())
        owner_share = (owner_hits / total_hits) if total_hits > 0 else 0.0

        segments.append(
            {
                "start_frame": frame_ids[i],
                "end_frame": frame_ids[best_j],
                "owner_player_id": best_owner,
                "length": length,
                "owner_share": owner_share,
            }
        )
        i = best_j + 1

    return segments
