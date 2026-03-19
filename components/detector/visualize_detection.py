from pathlib import Path
from detector import (
    Detector,
    get_video_players_detections,
    get_video_ball_detections,
    get_video_rim_detections,
    get_video_referee_detections,
    get_frame_number_detections,
    get_frame_pose_detections,
    match_poses_to_players,
    match_numbers_to_players,
)
from detector.number_recognizer_parseq import recognize_numbers_in_frame as recognize_numbers_parseq

import argparse
import cv2
import numpy as np
from skimage.feature import match_template
from tqdm.auto import tqdm
from detector.remove_bad_ball_detections import remove_bad_ball_detections

from court_detector.court_detector import CourtDetector
from court_detector.court_constants import CourtConstants
from common.classes import CourtType


def _draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    *,
    keypoint_color: tuple[int, int, int] = (0, 255, 0),
    limb_color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = 2,
    conf_threshold: float = 0.1,
) -> None:
    """Draw keypoints (K,3) and a simple skeleton on the frame."""
    if keypoints is None or keypoints.size == 0:
        return
    num_kp = keypoints.shape[0]

    for idx in range(num_kp):
        x, y, c = keypoints[idx]
        if c < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, keypoint_color, -1, lineType=cv2.LINE_AA)

    # COCO-style subset (works for common 17-kp layouts)
    skeleton = [
        (5, 6),  # shoulders
        (5, 7),
        (7, 9),  # left arm
        (6, 8),
        (8, 10),  # right arm
        (11, 12),  # hips
        (11, 13),
        (13, 15),  # left leg
        (12, 14),
        (14, 16),  # right leg
    ]
    for i, j in skeleton:
        if i >= num_kp or j >= num_kp:
            continue
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 < conf_threshold or c2 < conf_threshold:
            continue
        cv2.line(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            limb_color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def video_with_ball_bbox_yolo(
    input_path: str,
    output_path: str | None = None,
    conf_threshold: float = 0.3,
    draw_all: bool = False,
    roi_conf_threshold: float = 0.5,
    *,
    draw_player_bboxes: bool = False,
    draw_player_numbers: bool = False,
    draw_player_skeletons: bool = False,
) -> str:
    """
    Two passes: 1) detection (YOLO + template matching), 2) drawing with gap interpolation.
    If a frame has no bbox, it is interpolated along the line between previous and next frames with bbox.

    Args:
        input_path: path to input video.
        output_path: path to output video. If None — next to input as <stem>_ball_yolo.mp4.
        conf_threshold: confidence threshold for ball detection.
        draw_all: if True, draw all detections; otherwise only the best.
        roi_conf_threshold: unused (kept for compatibility).

    Returns:
        Path to the saved file.
    """
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_ball_yolo{p.suffix}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    detector = Detector()
    detections = detector.detect_video(input_path)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    color_player = (255, 165, 0)  # BGR cyan (players without number)
    color_player_with_number = (0, 255, 0)  # BGR green (players with assigned number)
    color_referee = (255, 0, 255)  # BGR magenta (referees)
    thickness = 2

    pbar2 = tqdm(total=total_frames, desc="Pass 2: draw", unit="frame")

    ball_detections = get_video_ball_detections(detections)
    player_detections = (
        get_video_players_detections(detections, conf_threshold=0.1)
        if (draw_player_bboxes or draw_player_numbers or draw_player_skeletons)
        else {}
    )
    referee_detections = get_video_referee_detections(detections, conf_threshold=0.1) if draw_player_numbers else {}
    rim_detections = get_video_rim_detections(detections, conf_threshold=0.1)

    # Homographies: frame normalized coords [0,1]^2 -> court normalized coords [-0.5,0.5]^2.
    court_constants = CourtConstants(CourtType.NBA)
    court_detector = CourtDetector()
    homographies, _, _, _ = court_detector.extract_homographies_from_video_v2(str(input_path), court_constants)

    kept_ball_by_frame = remove_bad_ball_detections(
        ball_detections,
        frame_size=(w, h),
        homographies=homographies,
        conf_threshold=conf_threshold,
    )

    # Directory for frames with player detections > 10 or ball detections > 1
    # many_players_dir = Path(input_path).parent / f"{Path(input_path).stem}_frames_many_players"
    # many_players_dir.mkdir(parents=True, exist_ok=True)
    # PLAYER_DETECTION_SAVE_THRESHOLD = 10
    # BALL_DETECTION_SAVE_THRESHOLD = 1

    max_player_detections = 0
    max_ball_detections = 0
    avg_players = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    try:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_label = f"frame: {i}"
            (tw, th), _ = cv2.getTextSize(frame_label, font, 0.8, 2)
            x0, y0 = 10, 30
            cv2.rectangle(frame, (x0 - 6, y0 - th - 8), (x0 + tw + 6, y0 + 6), (0, 0, 0), -1)
            cv2.putText(frame, frame_label, (x0, y0), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            players_in_frame = player_detections.get(i, [])
            balls_in_frame = ball_detections.get(i, [])
            kept_ball = kept_ball_by_frame.get(i)
            referees_in_frame = referee_detections.get(i, [])
            rims_in_frame = rim_detections.get(i, [])
            if draw_player_numbers:
                numbers_in_frame = get_frame_number_detections(detections[i], frame=None, conf_threshold=0.25)
                recognize_numbers_parseq(frame, numbers_in_frame, padding=5, ocr_conf_threshold=0.999)
                match_numbers_to_players(
                    {i: players_in_frame},
                    {i: numbers_in_frame},
                    {i: referees_in_frame},
                )

            if draw_player_skeletons:
                # Pose (YOLO-pose) -> match to our players -> store in player.skeleton
                poses_in_frame = get_frame_pose_detections(frame, conf_threshold=0.15)
                match_poses_to_players(players_in_frame, poses_in_frame, iou_threshold=0.3)

            def draw_detection(ball_detection, color):
                detection = ball_detection.bbox
                x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                if ball_detection.confidence is not None:
                    label = f"{ball_detection.confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 2), font, font_scale, (0, 0, 0), font_thickness)

            for ball_detection in balls_in_frame:
                draw_detection(ball_detection, (0, 0, 255))

            if kept_ball is not None:
                draw_detection(kept_ball, (0, 255, 0))

            if draw_player_bboxes or draw_player_numbers or draw_player_skeletons:
                for player_detection in players_in_frame:
                    detection = player_detection.bbox
                    x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
                    has_recognized_num = (
                        draw_player_numbers
                        and player_detection.number is not None
                        and player_detection.number.num is not None
                    )
                    color = color_player_with_number if has_recognized_num else color_player

                    if draw_player_bboxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    if (
                        draw_player_skeletons
                        and player_detection.skeleton is not None
                        and player_detection.skeleton.keypoints is not None
                    ):
                        _draw_skeleton(frame, player_detection.skeleton.keypoints)

                    if draw_player_numbers:
                        if has_recognized_num:
                            label = str(player_detection.number.num)
                        elif player_detection.confidence is not None:
                            label = f"{player_detection.confidence:.2f}"
                        else:
                            label = None
                        if label is not None and draw_player_bboxes:
                            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 2, y1 - 2), font, font_scale, (0, 0, 0), font_thickness)

            for referee_detection in referees_in_frame:
                detection = referee_detection.bbox
                x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, thickness)
                if referee_detection.confidence is not None:
                    label = f"{referee_detection.confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color_referee, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 2), font, font_scale, (0, 0, 0), font_thickness)

            # Draw rim bboxes (class_id 10) from raw detections
            for rim_det in rims_in_frame:
                x1, y1, x2, y2 = rim_det.get_bbox()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)

            num_players = len(players_in_frame)
            num_balls = len(balls_in_frame)
            max_player_detections = max(max_player_detections, num_players)
            max_ball_detections = max(max_ball_detections, num_balls)

            avg_players += len(players_in_frame)

            # if num_players > PLAYER_DETECTION_SAVE_THRESHOLD or num_balls > BALL_DETECTION_SAVE_THRESHOLD:
            #     frame_path = many_players_dir / f"frame_{i:06d}_players_{num_players}_balls_{num_balls}.jpg"
            #     cv2.imwrite(str(frame_path), frame)

            writer.write(frame)
            pbar2.update(1)
            # break
    finally:
        pbar2.close()
        cap.release()
        writer.release()

    # print(f"[YOLO] Ball detected on {detected_count} frames, interpolated gaps, written to {output_path}")
    print(f"Max players: {max_player_detections}")
    print(f"Max balls: {max_ball_detections}")
    print(f"Avg players: {avg_players / total_frames}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to output video")
    parser.add_argument("--conf", type=float, default=0.3, help="Ball confidence threshold")
    parser.add_argument("--draw-all", action="store_true", help="Draw all ball detections (unused; kept)")
    parser.add_argument("--draw-player-bboxes", action="store_true", help="Draw player bounding boxes")
    parser.add_argument("--draw-player-numbers", action="store_true", help="Run number recognition and draw labels")
    parser.add_argument("--draw-player-skeletons", action="store_true", help="Run pose detection and draw skeletons")
    args = parser.parse_args()

    video_with_ball_bbox_yolo(
        args.input,
        args.output,
        conf_threshold=args.conf,
        draw_all=args.draw_all,
        draw_player_bboxes=args.draw_player_bboxes,
        draw_player_numbers=args.draw_player_numbers,
        draw_player_skeletons=args.draw_player_skeletons,
    )


if __name__ == "__main__":
    main()
