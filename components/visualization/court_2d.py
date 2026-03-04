"""
2D court view renderer: projects meter coordinates onto a top-down court template.

Renders players and ball positions (in court meters) onto NBA/FIBA court images.
Center of court = (0, 0), right = +x.
"""

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from tqdm.auto import tqdm

from common.classes import CourtType
from common.classes import FrameDetections

_ASSETS_DIR = Path(__file__).parent / "assets"
PATH_VIA_LEAGUE = {
    CourtType.NBA: str(_ASSETS_DIR / "nba.png"),
    CourtType.FIBA: str(_ASSETS_DIR / "fiba.png"),
}

# Parameters used when saving base court (scale=10, padding=40)
_COURT_PADDING = 40

# Court dimensions in meters (length, width). Center = (0, 0), right = +x.
_COURT_METERS = {
    CourtType.NBA: (28.65, 15.24),
    CourtType.FIBA: (28.0, 15.0),
}


def _meters_to_norm(x_m: float, y_m: float, length_m: float, width_m: float) -> Tuple[float, float]:
    """Meters (center 0,0, right +x) → normalized [0, 1]."""
    x_norm = (x_m / length_m) + 0.5
    y_norm = (y_m / width_m) + 0.5
    return (x_norm, y_norm)


def _norm_to_pixel(x: float, y: float, width: int, height: int, padding: int = _COURT_PADDING) -> Tuple[int, int]:
    """Normalized [0, 1] → pixel coordinates on saved court."""
    px = int(round(padding + x * (width - 2 * padding)))
    py = int(round(padding + y * (height - 2 * padding)))
    return (px, py)


class Court2DView:
    """
    Renders a top-down 2D court view from meter coordinates.

    Draws team positions and ball on NBA/FIBA court templates.
    """

    _league: CourtType
    _base_court: np.ndarray

    def __init__(self, league: CourtType):
        self._league = league
        path = PATH_VIA_LEAGUE[league]
        self._base_court = cv2.imread(path)
        if self._base_court is None:
            raise FileNotFoundError(
                f"Court image not found: {path}. Add nba.png or fiba.png to components/visualization/assets/."
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
        Render 2D points of two teams and ball on the base court.

        Coordinates in meters: center = (0, 0), right = +x.
        team1_xy, team2_xy: arrays of shape (N, 2), (x_m, y_m).
        ball_xy: (x_m, y_m) or array of shape (2,).
        Colors in BGR (OpenCV).
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


def write_2d_court_video(
    detections: FrameDetections,
    output_path: str,
    league: CourtType,
    video_path: str,
) -> None:
    """Render 2D court view for each frame and write to video."""
    court_view = Court2DView(league)
    h, w = court_view._base_court.shape[:2]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_id in tqdm(range(1, total_frames + 1), desc="Writing 2D video"):
        players = detections.get(frame_id, [])
        team1_xy = []
        team2_xy = []
        for p in players:
            if p.court_position is None or p.team_id is None:
                continue
            x_m, y_m = p.court_position
            xy = np.array([[x_m, y_m]])
            if p.team_id == 0:
                team1_xy.append(xy)
            else:
                team2_xy.append(xy)

        team1_xy = np.vstack(team1_xy) if team1_xy else np.empty((0, 2))
        team2_xy = np.vstack(team2_xy) if team2_xy else np.empty((0, 2))
        ball_xy = (0.0, 0.0)

        frame = court_view.get_frame(team1_xy, team2_xy, ball_xy)
        out.write(frame)

    out.release()
