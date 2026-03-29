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
from tqdm.auto import tqdm
from detector.remove_bad_ball_detections import remove_bad_ball_detections
from common.distances import bbox_iou

from court_detector.court_detector import CourtDetector
from court_detector.court_constants import CourtConstants
from common.classes import CourtType
from ball_detector.detector import WASBBallDetector
from video_reader import VideoReader


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
    *,
    draw_player_bboxes: bool = False,
    draw_player_numbers: bool = False,
    draw_player_skeletons: bool = False,
    draw_pose_bboxes: bool = False,
) -> str:
    """
    Two passes: 1) detection (YOLO + template matching), 2) drawing with gap interpolation.
    If a frame has no bbox, it is interpolated along the line between previous and next frames with bbox.

    Args:
        input_path: path to input video.
        output_path: path to output video.

    Returns:
        Path to the saved file.
    """

    vr = VideoReader(input_path)

    fps = vr.get(cv2.CAP_PROP_FPS)
    w = int(vr.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = Detector()
    detections = detector.detect_video(vr)

    vr.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    color_player = (255, 165, 0)  # BGR cyan (players without number)
    color_player_with_number = (0, 255, 0)  # BGR green (players with assigned number)
    color_referee = (255, 0, 255)  # BGR magenta (referees)
    thickness = 2

    pbar2 = tqdm(total=total_frames, desc="Pass 2: draw", unit="frame")

    ball_detections = get_video_ball_detections(detections)
    player_detections = get_video_players_detections(detections, conf_threshold=0.1)

    referee_detections = get_video_referee_detections(detections, conf_threshold=0.1)
    rim_detections = get_video_rim_detections(detections, conf_threshold=0.1)

    # Homographies: frame normalized coords [0,1]^2 -> court normalized coords [-0.5,0.5]^2.
    court_constants = CourtConstants(CourtType.NBA)
    court_detector = CourtDetector()
    homographies, _, _ = court_detector.extract_homographies_from_video_v2(vr, court_constants)
    vr.set(cv2.CAP_PROP_POS_FRAMES, 0)

    kept_ball_by_frame = remove_bad_ball_detections(
        ball_detections,
        frame_size=(w, h),
        homographies=homographies,
    )
    interpolated_ball_by_frame = kept_ball_by_frame

    max_player_detections = 0
    max_ball_detections = 0
    avg_players = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    try:
        for i in range(total_frames):
            ret, frame = vr.read()
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
            interpolated_ball = interpolated_ball_by_frame.get(i)
            referees_in_frame = referee_detections.get(i, [])
            rims_in_frame = rim_detections.get(i, [])
            if draw_player_numbers:
                numbers_in_frame = get_frame_number_detections(detections[i])
                recognize_numbers_parseq(frame, numbers_in_frame, padding=5, ocr_conf_threshold=0.999)
                match_numbers_to_players(
                    {i: players_in_frame},
                    {i: numbers_in_frame},
                    {i: referees_in_frame},
                )

            if draw_player_skeletons:
                # Pose (YOLO-pose) -> match to our players -> store in player.skeleton
                poses_in_frame = get_frame_pose_detections(frame)
                match_poses_to_players(players_in_frame, poses_in_frame, iou_threshold=0.7)
                # Build best pose match per player index for optional pose-bbox drawing.
                pose_by_player_idx: dict[int, object] = {}
                for p_idx, player_detection in enumerate(players_in_frame):
                    if not player_detection.bbox:
                        continue
                    best_iou = 0.0
                    best_pose = None
                    for pose in poses_in_frame:
                        iou = bbox_iou(player_detection.bbox, pose.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_pose = pose
                    if best_pose is not None and best_iou >= 0.7:
                        pose_by_player_idx[p_idx] = best_pose
            else:
                pose_by_player_idx = {}

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
            elif interpolated_ball is not None:
                # Frames that exist only after interpolation are drawn in a separate color.
                draw_detection(interpolated_ball, (255, 0, 255))

            for p_idx, player_detection in enumerate(players_in_frame):
                if draw_pose_bboxes and draw_player_skeletons and p_idx in pose_by_player_idx:
                    detection = pose_by_player_idx[p_idx].bbox
                else:
                    detection = player_detection.bbox
                x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                has_recognized_num = (
                    draw_player_numbers
                    and player_detection.number is not None
                    and player_detection.number.num is not None
                )
                color = color_player_with_number if has_recognized_num else color_player

                if draw_player_bboxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    if getattr(player_detection, "is_dribble", False):
                        cx = (x1 + x2) // 2
                        top_y = max(0, y1 - 40)
                        cv2.arrowedLine(
                            frame,
                            (cx, top_y),
                            (cx, y1),
                            (0, 255, 255),  # bright yellow/cyan
                            3,
                            tipLength=0.3,
                        )

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

            # Draw rim bboxes (class_id 10) from raw detections
            for rim_det in rims_in_frame:
                x1, y1, x2, y2 = rim_det.get_bbox()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)

            num_players = len(players_in_frame)
            num_balls = len(balls_in_frame)
            max_player_detections = max(max_player_detections, num_players)
            max_ball_detections = max(max_ball_detections, num_balls)

            avg_players += len(players_in_frame)

            writer.write(frame)
            pbar2.update(1)
            # break
    finally:
        pbar2.close()
        vr.release()
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
    parser.add_argument("--draw-player-bboxes", action="store_true", help="Draw player bounding boxes")
    parser.add_argument("--draw-player-numbers", action="store_true", help="Run number recognition and draw labels")
    parser.add_argument("--draw-player-skeletons", action="store_true", help="Run pose detection and draw skeletons")
    parser.add_argument(
        "--draw-pose-bboxes",
        action="store_true",
        help="Draw pose-matched player bboxes instead of detector bboxes (requires --draw-player-skeletons)",
    )
    args = parser.parse_args()

    video_with_ball_bbox_yolo(
        args.input,
        args.output,
        draw_player_bboxes=args.draw_player_bboxes,
        draw_player_numbers=args.draw_player_numbers,
        draw_player_skeletons=args.draw_player_skeletons,
        draw_pose_bboxes=args.draw_pose_bboxes,
    )


if __name__ == "__main__":
    main()
