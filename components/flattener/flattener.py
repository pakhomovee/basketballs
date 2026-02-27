import os
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

from classes.classes import CourtType

_flattener_dir = Path(__file__).parent.resolve()
PATH_VIA_LEAGUE = {
    CourtType.NBA: str(_flattener_dir / "nba.png"),
    CourtType.FIBA: str(_flattener_dir / "fiba.png"),
}

# Параметры, с которыми был сохранён базовый корт (scale=10, padding=40)
_COURT_PADDING = 40

# Размеры корта в метрах (длина, ширина) для перевода координат. Центр поля = (0, 0), вправо = +x.
_COURT_METERS = {
    CourtType.NBA: (28.65, 15.24),  # length, width
    CourtType.FIBA: (28.0, 15.0),
}


def _meters_to_norm(x_m: float, y_m: float, length_m: float, width_m: float) -> Tuple[float, float]:
    """Координаты в метрах (центр 0,0, вправо +x) → нормализованные [0, 1]."""
    x_norm = (x_m / length_m) + 0.5  # -L/2..+L/2 → 0..1
    y_norm = (y_m / width_m) + 0.5  # -W/2..+W/2 → 0..1
    return (x_norm, y_norm)


def _norm_to_pixel(x: float, y: float, width: int, height: int, padding: int = _COURT_PADDING) -> Tuple[int, int]:
    """Нормализованные координаты [0, 1] в пиксели на сохранённом корте."""
    px = int(round(padding + x * (width - 2 * padding)))
    py = int(round(padding + y * (height - 2 * padding)))
    return (px, py)


class Flattener:
    _league: CourtType
    _base_court: np.ndarray

    def __init__(self, league: CourtType):
        self._league = league
        path = PATH_VIA_LEAGUE[league]
        self._base_court = cv2.imread(path)
        if self._base_court is None:
            raise FileNotFoundError(
                f"Court image not found: {path}. "
                "Add nba.png or fiba.png to components/flattener/."
            )

    def get_frame(
        self,
        team1_xy: np.ndarray,
        team2_xy: np.ndarray,
        ball_xy: Union[Tuple[float, float], np.ndarray],
        team1_color: Tuple[int, int, int] = (0, 0, 255),
        team2_color: Tuple[int, int, int] = (255, 0, 0),
        ball_color: Tuple[int, int, int] = (0, 165, 255),
        player_radius: int = 12,
        ball_radius: int = 8,
    ) -> np.ndarray:
        """
        Отрисовать 2D точки двух команд и мяча на базовом корте.

        Координаты в метрах: центр поля = (0, 0), вправо = положительная ось x.
        team1_xy, team2_xy: массивы формы (N, 2), (x_m, y_m).
        ball_xy: (x_m, y_m) или массив формы (2,).
        Цвета в BGR (OpenCV).
        """
        frame = self._base_court.copy()
        h, w = frame.shape[:2]
        pad = _COURT_PADDING
        length_m, width_m = _COURT_METERS[self._league]

        def to_pixel(x_m: float, y_m: float) -> Tuple[int, int]:
            y_m = -y_m
            xn, yn = _meters_to_norm(x_m, y_m, length_m, width_m)
            return _norm_to_pixel(xn, yn, w, h, pad)

        def draw_points(xy: np.ndarray, color: Tuple[int, int, int], radius: int) -> None:
            pts = np.atleast_2d(xy)
            for i in range(pts.shape[0]):
                x_m, y_m = float(pts[i, 0]), float(pts[i, 1])
                cx, cy = to_pixel(x_m, y_m)
                cv2.circle(frame, (cx, cy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), radius, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        draw_points(team1_xy, team1_color, player_radius)
        draw_points(team2_xy, team2_color, player_radius)

        ball = np.atleast_1d(ball_xy)
        if ball.size >= 2:
            bx, by = float(ball[0]), float(ball[1])
            bcx, bcy = to_pixel(bx, by)
            cv2.circle(frame, (bcx, bcy), ball_radius, ball_color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (bcx, bcy), ball_radius, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        return frame


if __name__ == "__main__":
    # Пробный запуск get_frame: координаты в метрах, центр (0,0), вправо +x
    _dir = os.path.dirname(os.path.abspath(__file__))
    _flattener = Flattener(CourtType.NBA)
    _team1 = np.array(
        [[_COURT_METERS[CourtType.NBA][0] / 2, _COURT_METERS[CourtType.NBA][1] / 2], [-10.0, 2.0], [-10.0, -2.0]]
    )  # слева
    _team2 = np.array([[10.0, 0.0], [10.0, 2.0], [10.0, -2.0]])  # справа
    _ball = (0.0, 0.0)  # центр поля
    _frame = _flattener.get_frame(_team1, _team2, _ball)
    _out = os.path.join(_dir, "nba_sample.png")
    cv2.imwrite(_out, _frame)
    print(f"Saved: {_out}")
