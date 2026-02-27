from typing import Iterable, Optional, Tuple
import os
import sys

import cv2
import numpy as np
import supervision as sv

_THIS_DIR = os.path.dirname(__file__)
_COMPONENTS_DIR = os.path.dirname(_THIS_DIR)
if _COMPONENTS_DIR not in sys.path:
    sys.path.append(_COMPONENTS_DIR)

# from court_detector.court_detector import CourtDetector
# from classes.classes import League

# from court_detector.court_detector import CourtDetector
from sports.basketball.config import CourtConfiguration, League, MeasurementUnit
from sports.basketball.annotators import draw_court, draw_points_on_court

def save_base_court(
    court_array_path: str,
    league: League
) -> None:
    """
    Пример: сгенерировать короткое видео с одним бегущим игроком.

    Видео рисуется на геометрически точном корте через draw_court +
    draw_points_on_court.

    Если передан court_array_path, базовый корт (np.ndarray) сохраняется
    в этот файл через np.save(); загрузить обратно: np.load(path).
    """
    config = CourtConfiguration(
        league=league,
        measurement_unit=MeasurementUnit.FEET,
    )

    # базовый корт рисуем один раз (те же параметры: scale=10, padding=40)
    base_court = draw_court(config, scale=10, padding=40)
    height, width = base_court.shape[:2]

    if court_array_path is not None:
        cv2.imwrite(court_array_path, base_court)
        # np.save(court_array_path, base_court)
    return

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        for t in range(num_frames):
            # один игрок, бегущий вдоль длины площадки
            x = config.court_length * (t / max(1, num_frames - 1))
            y = config.court_width / 2.0

            pts = np.array([[x, y]], dtype=float)

            frame = base_court.copy()
            frame = draw_points_on_court(
                config=config,
                xy=pts,
                fill_color=sv.Color.from_hex("#ffff00"),
                edge_color=sv.Color.BLACK,
                court=frame,
            )

            writer.write(frame)
    finally:
        writer.release()

# Пример: сохранить корт в .npy и снять видео
# Загрузить корт обратно: base_court = np.load("base_court.npy")
# render_example_video(
#     # output_path="players2.mp4",
#     court_array_path="base_court.npy",
# )
save_base_court("./nba.png", League.NBA)
