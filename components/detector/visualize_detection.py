from pathlib import Path
from detector import Detector

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
    Два прохода: 1) детекция (YOLO + template matching), 2) отрисовка с интерполяцией пропусков.
    Если в кадре нет bbox, он интерполируется по прямой между предыдущим и следующим кадром с bbox.

    Args:
        input_path: путь к входному видео.
        output_path: путь к выходному видео. Если None — рядом с входным, <stem>_ball_yolo.mp4.
        conf_threshold: порог уверенности для детекции мяча.
        draw_all: если True, рисуются все детекции; иначе только лучшая.
        roi_conf_threshold: не используется (оставлен для совместимости).

    Returns:
        Путь к сохранённому файлу.
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
    color_yolo = (0, 165, 255)  # BGR оранжевый (мяч)
    color_player = (255, 165, 0)  # BGR голубой (игроки, только YOLO)
    thickness = 2

    pbar2 = tqdm(total=total_frames, desc="Pass 2: draw", unit="frame")
    index = 0
    try:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Мяч
            while index < len(detections) and detections[index].frame_id < i:
                index += 1
            if index < len(detections):
                d = detections[index].detections
                for detection in d:
                    if detection.confidence < 0.5:
                        continue
                    if detection.class_id == 0:
                        cv2.rectangle(
                            frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color_yolo, thickness
                        )
                    elif detection.class_id >= 2 and detection.class_id <= 8:
                        cv2.rectangle(
                            frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color_player, thickness
                        )

            writer.write(frame)
            pbar2.update(1)
    finally:
        pbar2.close()
        cap.release()
        writer.release()

    # print(f"[YOLO] Ball detected on {detected_count} frames, interpolated gaps, written to {output_path}")
    return output_path


video_with_ball_bbox_yolo("./img1-3.mp4", "./matching16.mp4")
