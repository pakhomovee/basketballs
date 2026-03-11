"""
Annotation tool to correct team clustering predictions and create ground truth.

Usage:
    python -m components.team_clustering.annotate [--predictions PATH] [--output PATH]

Controls:
    n / SPACE   - Next image
    p           - Previous image
    x           - Skip image (exclude from ground truth)
    r           - Toggle remove mode (then left-click to remove bbox)
    Left-click  - Cycle team (0 -> 1 -> any -> 0), or remove bbox if in remove mode
    Right-click - Remove bbox (FP detection; use r+click on Mac)
    s           - Save ground truth
    q / ESC     - Quit
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# Team colors (BGR): team 0 = blue, team 1 = red, any = yellow
TEAM_COLORS = [(255, 100, 50), (50, 50, 255)]  # BGR for 0, 1
TEAM_ANY = "any"
TEAM_ANY_COLOR = (0, 200, 255)  # BGR yellow
UNASSIGNED_COLOR = (128, 128, 128)


def _point_in_bbox(x: int, y: int, bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox[:4]
    return x1 <= x <= x2 and y1 <= y <= y2


def _find_clicked_bbox(x: int, y: int, bboxes: list[list[int]]) -> int | None:
    """Return index of bbox containing (x, y), or None. Prefer smaller bboxes (closer)."""
    candidates = []
    for i, bbox in enumerate(bboxes):
        if _point_in_bbox(x, y, bbox):
            x1, y1, x2, y2 = bbox[:4]
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, i))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def draw_annotations(
    frame: np.ndarray,
    bboxes: list[list[int]],
    team_labels: list[int | None],
    highlight_idx: int | None = None,
    skipped: bool = False,
) -> np.ndarray:
    """Draw bboxes with team colors. Returns a copy."""
    out = frame.copy()
    if skipped:
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (out.shape[1], out.shape[0]), (0, 0, 128), -1)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
        cv2.putText(out, "SKIPPED", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    for i, (bbox, label) in enumerate(zip(bboxes, team_labels)):
        x1, y1, x2, y2 = map(int, bbox[:4])
        if label == TEAM_ANY:
            color = TEAM_ANY_COLOR
            label_str = "any"
        elif label is not None and 0 <= label < len(TEAM_COLORS):
            color = TEAM_COLORS[label]
            label_str = str(label)
        else:
            color = UNASSIGNED_COLOR
            label_str = "?"
        thickness = 3 if i == highlight_idx else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(out, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def run_annotator(
    predictions_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Interactive annotation tool. Load predictions, allow corrections, save ground truth.

    Args:
        predictions_path: JSON from benchmark.py (image_path, bboxes, team_labels).
        output_path: Where to save corrected ground truth.
    """
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)

    with open(predictions_path) as f:
        data = json.load(f)

    if not data:
        print("No predictions to annotate.")
        return

    _root = Path(__file__).resolve().parent.parent.parent

    # Work on a copy so we can save at any time
    items = [dict(d) for d in data]
    for item in items:
        item.setdefault("skipped", False)
    idx = 0
    remove_mode = False
    window_name = "Team clustering annotation"

    scale_factor = 1.0

    def _resolve_image_path(path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = _root / p
        return p

    def refresh():
        item = items[idx]
        img_path = _resolve_image_path(item["image_path"])
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Cannot load {img_path!s}")
            return None
        bboxes = item["bboxes"]
        labels = list(item["team_labels"])
        return draw_annotations(frame, bboxes, labels, skipped=item.get("skipped", False))

    def show():
        nonlocal scale_factor
        disp = refresh()
        if disp is not None:
            h, w = disp.shape[:2]
            max_h = 900
            if h > max_h:
                scale_factor = max_h / h
                disp = cv2.resize(disp, (int(w * scale_factor), int(h * scale_factor)))
            else:
                scale_factor = 1.0
            title = f"{window_name} [REMOVE MODE]" if remove_mode else window_name
            cv2.setWindowTitle(window_name, title)
            cv2.imshow(window_name, disp)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        lambda event, x, y, *_args: on_mouse(event, x, y),
    )

    def on_mouse(event: int, x: int, y: int) -> None:
        nonlocal scale_factor, remove_mode
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        item = items[idx]
        bboxes = item["bboxes"]
        labels = item["team_labels"]
        x_orig = int(x / scale_factor)
        y_orig = int(y / scale_factor)
        clicked = _find_clicked_bbox(x_orig, y_orig, bboxes)
        if clicked is None:
            return
        do_remove = remove_mode or event == cv2.EVENT_RBUTTONDOWN
        if do_remove:
            bboxes.pop(clicked)
            labels.pop(clicked)
            item["bboxes"] = bboxes
            item["team_labels"] = labels
        else:
            old = labels[clicked]
            # Cycle: 0 -> 1 -> any -> 0
            if old == 0:
                labels[clicked] = 1
            elif old == 1:
                labels[clicked] = TEAM_ANY
            else:
                labels[clicked] = 0
            item["team_labels"] = labels
        show()

    show()

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("n"), ord(" ")):
            idx = min(idx + 1, len(items) - 1)
            show()
        elif k == ord("p"):
            idx = max(idx - 1, 0)
            show()
        elif k == ord("x"):
            items[idx]["skipped"] = not items[idx].get("skipped", False)
            show()
        elif k == ord("r"):
            remove_mode = not remove_mode
            show()
        elif k == ord("s"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            to_save = [
                {k: v for k, v in i.items() if k != "skipped"}
                for i in items if not i.get("skipped", False)
            ]
            with open(output_path, "w") as f:
                json.dump(to_save, f, indent=2)
            print(f"Saved {len(to_save)} images to ground truth ({len(items) - len(to_save)} skipped)")

    cv2.destroyAllWindows()


def main() -> None:
    _root = Path(__file__).resolve().parent.parent.parent
    run_annotator(
        predictions_path=_root / "dataset" / "team_clustering_benchmark" / "predictions.json",
        output_path=_root / "dataset" / "team_clustering_benchmark" / "ground_truth.json",
    )


if __name__ == "__main__":
    main()
