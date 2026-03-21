from __future__ import annotations

from common.classes.ball import Ball


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def linear_interpolate_ball_detections(
    ball_detections: dict[int, Ball],
    *,
    max_gap: int = 50,
    interpolated_confidence: float = 0.0,
) -> dict[int, Ball]:
    """
    Fill missing frames between known ball detections with linear interpolation.

    Input format matches remove_bad_ball_detections.py output:
        frame_id -> Ball

    Interpolates bbox corners [x1, y1, x2, y2] directly.
    Original detections are preserved; only missing frames are inserted.

    Args:
        ball_detections: sparse trajectory as dict[frame_id, Ball].
        max_gap: interpolate only if gap between known frames is <= max_gap.
        interpolated_confidence: confidence assigned to interpolated Ball.

    Returns:
        Dense dict[frame_id, Ball] containing originals + interpolated frames.
    """
    if not ball_detections:
        return {}

    sorted_frames = sorted(ball_detections.keys())
    out: dict[int, Ball] = {f: ball_detections[f] for f in sorted_frames}

    for i in range(len(sorted_frames) - 1):
        f1 = sorted_frames[i]
        f2 = sorted_frames[i + 1]
        gap = f2 - f1 - 1
        if gap <= 0:
            continue
        if gap > max_gap:
            continue

        b1 = ball_detections[f1]
        b2 = ball_detections[f2]
        if len(b1.bbox) < 4 or len(b2.bbox) < 4:
            continue

        x1a, y1a, x2a, y2a = b1.bbox[:4]
        x1b, y1b, x2b, y2b = b2.bbox[:4]

        for k in range(1, gap + 1):
            t = k / float(gap + 1)
            frame_id = f1 + k

            x1 = int(round(_lerp(x1a, x1b, t)))
            y1 = int(round(_lerp(y1a, y1b, t)))
            x2 = int(round(_lerp(x2a, x2b, t)))
            y2 = int(round(_lerp(y2a, y2b, t)))

            # Ensure valid bbox order after rounding.
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1

            out[frame_id] = Ball(
                bbox=[x1, y1, x2, y2],
                confidence=interpolated_confidence,
            )

    return dict(sorted(out.items(), key=lambda kv: kv[0]))
