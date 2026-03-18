"""
Лёгкая обёртка над YOLO-pose моделью для детекции поз игроков на баскетбольных видео.

Этот модуль специально сделан самодостаточным и НЕ изменяет существующий код проекта.
Его можно импортировать и использовать независимо от основного пайплайна детекции.

Пример:
    from detector.pose_detector import PoseDetector, detect_video_poses

    poses_by_frame = detect_video_poses("input.mp4")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from common.distances import bbox_iou
from detector import (
    Detector,
    get_video_players_detections,
    get_video_ball_detections,
    get_video_rim_detections,
)


@dataclass
class PoseDetection:
    """
    Одна детекция позы (для одного человека) на кадре.

    Поля:
        bbox: [x1, y1, x2, y2] в пикселях.
        keypoints: np.ndarray формы (K, 3) с (x, y, confidence) для каждой ключевой точки.
        score: уверенность детекции (confidence).
        class_id: id класса из pose-модели (обычно 0 для человека).
    """

    bbox: List[float] = field(default_factory=list)
    keypoints: np.ndarray | None = None
    score: float | None = None
    class_id: int | None = None


FramePoses = Dict[int, List[PoseDetection]]  # frame_id -> list of poses


class PoseDetector:
    """
    Детектор поз игроков на базе YOLO-pose.

    Модель загружается один раз и применяется покадрово.
    По умолчанию ожидается checkpoint pose-модели в папке models/ репозитория,
    но можно передать свой путь.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.1,
    ):
        """
        Args:
            model_path: путь до YOLO-pose checkpoint. Если None, используется
                <repo_root>/models/yolov8n-pose.pt по умолчанию.
            conf_threshold: минимальный confidence для сохранения pose-детекций.
        """
        repo_root = Path(__file__).resolve().parent.parent.parent
        default_model = repo_root / "models" / "yolov8m-pose.pt"
        self.model_path = Path(model_path) if model_path is not None else default_model
        self.conf_threshold = conf_threshold
        self.model = YOLO(str(self.model_path))

    def detect_frame(self, frame: np.ndarray) -> List[PoseDetection]:
        """
        Запустить pose-детекцию на одном кадре.

        Returns:
            Список PoseDetection для этого кадра.
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        poses: List[PoseDetection] = []

        # boxes.xyxy: (N, 4), boxes.conf: (N,), boxes.cls: (N,)
        # keypoints.xy: (N, K, 2), keypoints.conf: (N, K)
        boxes = getattr(results, "boxes", None)
        keypoints = getattr(results, "keypoints", None)

        if boxes is None or keypoints is None:
            return poses

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        k_xy = keypoints.xy.cpu().numpy()  # (N, K, 2)
        k_conf = keypoints.conf.cpu().numpy()  # (N, K)

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            score = float(confs[i])
            class_id = int(clses[i])
            # Объединяем координаты и confidence в (K, 3)
            kp_xy = k_xy[i]  # (K, 2)
            kp_c = k_conf[i][:, None]  # (K, 1)
            kp = np.concatenate([kp_xy, kp_c], axis=1)  # (K, 3)
            poses.append(
                PoseDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=kp,
                    score=score,
                    class_id=class_id,
                )
            )

        return poses

    def detect_video(self, video_path: str) -> FramePoses:
        """
        Запустить pose-детекцию на всех кадрах видео.

        Args:
            video_path: путь до входного видео.

        Returns:
            Словарь frame_id -> list[PoseDetection].
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        poses_by_frame: FramePoses = {}

        try:
            for frame_id in tqdm(range(frame_count), desc="PoseDetector", unit="frame"):
                ret, frame = cap.read()
                if not ret:
                    break
                poses_by_frame[frame_id] = self.detect_frame(frame)
        finally:
            cap.release()

        return poses_by_frame


def detect_video_poses(
    video_path: str,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.25,
) -> FramePoses:
    """
    Удобная обёртка: прогнать YOLO-pose по видео и вернуть позы по кадрам.

    Это тонкая обёртка вокруг PoseDetector для быстрого запуска.
    Функция самодостаточная и не меняет существующий код проекта.
    """
    detector = PoseDetector(model_path=model_path, conf_threshold=conf_threshold)
    return detector.detect_video(video_path)


def _draw_pose_skeleton(
    frame: np.ndarray,
    pose: PoseDetection,
    *,
    keypoint_color: Tuple[int, int, int] = (0, 255, 0),
    limb_color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = 2,
    conf_threshold: float = 0.3,
) -> None:
    """
    Нарисовать ключевые точки и простой "скелет" на кадре для одного PoseDetection.

    Точная связность зависит от порядка keypoints в YOLO-pose; здесь используется
    небольшой COCO-подобный набор связей, который обычно подходит для отладки.
    """
    if pose.keypoints is None or pose.keypoints.size == 0:
        return

    kps = pose.keypoints  # (K, 3) -> (x, y, c)
    num_kp = kps.shape[0]

    # Рисуем точки
    for idx in range(num_kp):
        x, y, c = kps[idx]
        if c < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, keypoint_color, -1, lineType=cv2.LINE_AA)

    # Простые связи "скелета" (для частого 17-kp порядка; с защитой по индексам)
    skeleton: Iterable[Tuple[int, int]] = [
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
        x1, y1, c1 = kps[i]
        x2, y2, c2 = kps[j]
        if c1 < conf_threshold or c2 < conf_threshold:
            continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), limb_color, thickness, lineType=cv2.LINE_AA)


def write_video_with_poses(
    input_video_path: str,
    output_video_path: str | None = None,
    *,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.25,
) -> str:
    """
    Прогнать YOLO-pose по видео и сохранить новое видео с отрисованными позами.

    Высокоуровневая обёртка:
      1) Открывает входное видео
      2) Запускает PoseDetector покадрово
      3) Рисует keypoints + "скелет" для каждого найденного человека
      4) Сохраняет результат в новый файл

    Returns:
        Путь к сохранённому видео.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video_path is None:
        p = Path(input_video_path)
        output_video_path = str(p.parent / f"{p.stem}_poses{p.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    detector = PoseDetector(model_path=model_path, conf_threshold=conf_threshold)

    try:
        frame_id = 0
        # Кол-во кадров может быть неизвестно/0, поэтому tqdm без total
        for _ in tqdm(iter(int, 1), desc="Pose video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            poses = detector.detect_frame(frame)
            for pose in poses:
                _draw_pose_skeleton(frame, pose)

            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    return output_video_path


def write_video_with_player_poses(
    input_video_path: str,
    output_video_path: str | None = None,
    *,
    pose_model_path: str | Path | None = None,
    pose_conf_threshold: float = 0.25,
    player_conf_threshold: float = 0.1,
) -> str:
    """
    Сначала детектим игроков, затем применяем YOLO-pose только на кропах игроков.

    Пайплайн:
      1) Detector() -> detect_video(input_video_path) -> VideoDetections
      2) get_video_players_detections(...) -> игроки по кадрам
      3) Для каждого кадра: кроп по bbox игрока и запуск PoseDetector на кропе
      4) Перенос keypoints обратно в координаты исходного кадра и отрисовка "скелета"

    Это даёт позы только для игроков и уменьшает работу pose‑модели на фоне.
    """
    # 1) Получаем детекции игроков по всему видео
    det = Detector()
    video_detections = det.detect_video(input_video_path)
    players_detections = get_video_players_detections(video_detections, conf_threshold=player_conf_threshold)

    # 2) Готовим ввод/вывод видео
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video_path is None:
        p = Path(input_video_path)
        output_video_path = str(p.parent / f"{p.stem}_player_poses{p.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    pose_detector = PoseDetector(model_path=pose_model_path, conf_threshold=pose_conf_threshold)

    try:
        frame_id = 0
        for _ in tqdm(iter(int, 1), desc="Player pose video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            players_in_frame = players_detections.get(frame_id, [])

            for player in players_in_frame:
                if not player.bbox:
                    continue
                x1, y1, x2, y2 = map(int, player.bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                poses = pose_detector.detect_frame(crop)
                # Переносим keypoints в координаты полного кадра и рисуем
                for pose in poses:
                    if pose.keypoints is not None and pose.keypoints.size > 0:
                        pose.keypoints[:, 0] += x1
                        pose.keypoints[:, 1] += y1
                    _draw_pose_skeleton(frame, pose)

            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    return output_video_path


def _match_poses_to_players(
    players,
    poses: List[PoseDetection],
    iou_threshold: float = 0.3,
):
    """
    Match YOLO-pose detections to existing Player objects by IoU of bboxes.

    Returns:
        dict[player] -> PoseDetection
    """
    matched: list[tuple] = []
    for player in players:
        if not player.bbox:
            continue
        best_iou = 0.0
        best_pose: PoseDetection | None = None
        for pose in poses:
            iou = bbox_iou(player.bbox, pose.bbox)
            if iou > best_iou:
                best_iou = iou
                best_pose = pose
        if best_pose is not None and best_iou >= iou_threshold:
            matched.append((player, best_pose))
    return matched


def _hand_centers_from_pose(
    pose: PoseDetection,
    *,
    left_wrist_idx: int = 9,
    right_wrist_idx: int = 10,
    conf_threshold: float = 0.1,
) -> List[Tuple[float, float]]:
    """
    Return list of hand points (left/right wrists) for pose.

    Каждая рука рассматривается отдельно; может вернуть 0, 1 или 2 точек.
    """
    if pose.keypoints is None or pose.keypoints.size == 0:
        return []
    kps = pose.keypoints
    num_kp = kps.shape[0]

    points: List[Tuple[float, float]] = []
    for idx in (left_wrist_idx, right_wrist_idx):
        if idx < num_kp:
            x, y, c = kps[idx]
            if c >= conf_threshold:
                points.append((float(x), float(y)))
    return points


def write_video_with_ball_handler_poses(
    input_video_path: str,
    output_video_path: str | None = None,
    *,
    pose_model_path: str | Path | None = None,
    pose_conf_threshold: float = 0.1,
    player_conf_threshold: float = 0.1,
    max_hand_ball_dist: float = 30.0,
) -> str:
    """
    Detect players + ball with your Detector, run YOLO-pose on full frames,
    матчить позы к игрокам и подсвечивать игрока с мячом стрелочкой.

    «Игрок с мячом» определяется как тот, чьи руки (средняя точка запястий)
    ближе всего к центру bbox мяча.
    """
    # 1) Детекции игроков и мяча по всему видео
    det = Detector()
    video_detections = det.detect_video(input_video_path)
    players_detections = get_video_players_detections(video_detections, conf_threshold=player_conf_threshold)
    ball_detections = get_video_ball_detections(video_detections)
    rim_detections = get_video_rim_detections(video_detections, conf_threshold=0.1)

    # 2) Подготовка видео I/O
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video_path is None:
        p = Path(input_video_path)
        output_video_path = str(p.parent / f"{p.stem}_ball_handler_poses{p.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    pose_detector = PoseDetector(model_path=pose_model_path, conf_threshold=pose_conf_threshold)

    arrow_color = (0, 0, 255)  # red
    arrow_thickness = 3

    try:
        frame_id = 0
        for _ in tqdm(iter(int, 1), desc="Ball handler pose video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            players_in_frame = players_detections.get(frame_id, [])
            balls_in_frame = ball_detections.get(frame_id, [])
            rims_in_frame = rim_detections.get(frame_id, [])

            poses = pose_detector.detect_frame(frame)
            matched = _match_poses_to_players(players_in_frame, poses)

            # Рисуем скелеты и рамки игроков
            for player, pose in matched:
                _draw_pose_skeleton(frame, pose)
                if player.bbox:
                    x1, y1, x2, y2 = map(int, player.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

            # Определяем игрока с мячом
            ball_center: Tuple[float, float] | None = None
            if balls_in_frame:
                # Берём самый уверенный мяч
                best_ball = max(balls_in_frame, key=lambda b: b.confidence or 0.0)
                bx1, by1, bx2, by2 = best_ball.bbox
                ball_center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                cv2.circle(frame, (int(ball_center[0]), int(ball_center[1])), 6, (0, 165, 255), -1)

            best_player_for_ball = None
            best_dist = float("inf")
            if ball_center is not None:
                bx, by = ball_center
                for player, pose in matched:
                    hand_points = _hand_centers_from_pose(pose)
                    if not hand_points:
                        continue
                    for hx, hy in hand_points:
                        dist2 = (hx - bx) ** 2 + (hy - by) ** 2
                        if dist2 < best_dist:
                            best_dist = dist2
                            best_player_for_ball = player

            # Рисуем стрелку над игроком с мячом только если руки достаточно близко к мячу
            if best_player_for_ball is not None and best_player_for_ball.bbox and best_dist <= max_hand_ball_dist**2:
                x1, y1, x2, y2 = map(int, best_player_for_ball.bbox)
                cx = (x1 + x2) // 2
                top_y = max(0, y1 - 40)
                # Стрелка вниз к игроку
                cv2.arrowedLine(
                    frame,
                    (cx, top_y),
                    (cx, y1),
                    arrow_color,
                    arrow_thickness,
                    tipLength=0.3,
                )

            # Рисуем rim (класс 10) из детектора
            for rim_det in rims_in_frame:
                rx1, ry1, rx2, ry2 = rim_det.get_bbox()
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    return output_video_path


def write_video_with_poses_rtmpose(
    input_video_path: str,
    output_video_path: str | None = None,
    *,
    model: str = "rtmpose-m_8xb256-420e_coco-256x192",
    device: str | None = None,
) -> str:
    """
    Эксперимент: использовать RTMPose (MMPose) вместо YOLO-pose по полному кадру.

    Требуется установленный mmpose (dev-1.x) и соответствующая модель RTMPose.
    Используется высокоуровневый MMPoseInferencer и его визуализация.
    """
    from mmpose.apis import MMPoseInferencer

    inferencer = MMPoseInferencer(model, device=device)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video_path is None:
        p = Path(input_video_path)
        output_video_path = str(p.parent / f"{p.stem}_poses_rtmpose{p.suffix}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    try:
        frame_id = 0
        for _ in tqdm(iter(int, 1), desc="RTMPose video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            # RTMPose ожидает RGB; конвертируем из BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # inferencer возвращает генератор; берём первый результат
            result_gen = inferencer(rgb, return_vis=True, show=False)
            result = next(result_gen)
            vis_list = result.get("visualization", None)
            if not vis_list:
                # если ничего не найдено — оставляем исходный кадр
                writer.write(frame)
            else:
                vis = vis_list[0]
                # визуализация в RGB; конвертируем обратно в BGR при необходимости
                if vis.shape[2] == 3:
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                else:
                    vis_bgr = vis
                # подстраховка: ресайз, если inferencer изменил размер
                if vis_bgr.shape[1] != w or vis_bgr.shape[0] != h:
                    vis_bgr = cv2.resize(vis_bgr, (w, h))
                writer.write(vis_bgr)

            frame_id += 1
    finally:
        cap.release()
        writer.release()

    return output_video_path


write_video_with_ball_handler_poses("test_nba3.mp4", "megakek4.mp4")
