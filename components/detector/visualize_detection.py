from pathlib import Path
from detector import (
    Detector,
    get_video_players_detections,
    get_video_ball_detections,
    get_video_referee_detections,
    get_frame_number_detections,
    match_numbers_to_players,
)
from detector.number_recognizer_parseq import recognize_numbers_in_frame as recognize_numbers_parseq

import cv2
import numpy as np
from skimage.feature import match_template
from tqdm.auto import tqdm


def video_with_ball_bbox_yolo(
    input_path: str,
    output_path: str | None = None,
    conf_threshold: float = 0.5,
    draw_all: bool = False,
    roi_conf_threshold: float = 0.5,
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
    color_ball = (0, 165, 255)  # BGR orange (ball)
    color_player = (255, 165, 0)  # BGR cyan (players without number)
    color_player_with_number = (0, 255, 0)  # BGR green (players with assigned number)
    color_referee = (255, 0, 255)  # BGR magenta (referees)
    thickness = 2

    pbar2 = tqdm(total=total_frames, desc="Pass 2: draw", unit="frame")

    player_detections = get_video_players_detections(detections, conf_threshold=0.1)
    ball_detections = get_video_ball_detections(detections)
    referee_detections = get_video_referee_detections(detections, conf_threshold=0.1)

    # Directory for frames with player detections > 10 or ball detections > 1
    many_players_dir = Path(input_path).parent / f"{Path(input_path).stem}_frames_many_players"
    many_players_dir.mkdir(parents=True, exist_ok=True)
    PLAYER_DETECTION_SAVE_THRESHOLD = 10
    BALL_DETECTION_SAVE_THRESHOLD = 1

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

            players_in_frame = player_detections.get(i, [])
            balls_in_frame = ball_detections.get(i, [])
            referees_in_frame = referee_detections.get(i, [])
            numbers_in_frame = get_frame_number_detections(detections[i], frame=None, conf_threshold=0.25)
            recognize_numbers_parseq(frame, numbers_in_frame, padding=5, ocr_conf_threshold=0.999)
            match_numbers_to_players(
                {i: players_in_frame},
                {i: numbers_in_frame},
                {i: referees_in_frame},
            )

            for ball_detection in balls_in_frame:
                detection = ball_detection.bbox
                x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, thickness)
                if ball_detection.confidence is not None:
                    label = f"{ball_detection.confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color_ball, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 2), font, font_scale, (0, 0, 0), font_thickness)

            for player_detection in players_in_frame:
                detection = player_detection.bbox
                x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
                has_recognized_num = player_detection.number is not None and player_detection.number.num is not None
                color = color_player_with_number if has_recognized_num else color_player
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                if has_recognized_num:
                    label = str(player_detection.number.num)
                elif player_detection.confidence is not None:
                    label = f"{player_detection.confidence:.2f}"
                else:
                    label = None
                if label is not None:
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

            num_players = len(players_in_frame)
            num_balls = len(balls_in_frame)
            max_player_detections = max(max_player_detections, num_players)
            max_ball_detections = max(max_ball_detections, num_balls)

            avg_players += len(players_in_frame)

            if num_players > PLAYER_DETECTION_SAVE_THRESHOLD or num_balls > BALL_DETECTION_SAVE_THRESHOLD:
                frame_path = many_players_dir / f"frame_{i:06d}_players_{num_players}_balls_{num_balls}.jpg"
                cv2.imwrite(str(frame_path), frame)

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


video_with_ball_bbox_yolo("./test_nba3.mp4", "./test_kek3.mp4")
