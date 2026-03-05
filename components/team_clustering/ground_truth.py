import collections
import cv2

from common.classes.player import Player, PlayersDetections


def load_ground_truth(gt_path: str, video_path: str) -> PlayersDetections:
    """
    Load ground truth with normalized coordinates (x_center, y_center, w, h)
    and convert them to absolute pixel bboxes in xyxy format.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    gt_data: PlayersDetections = collections.defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x_center_norm = float(parts[2])
            y_center_norm = float(parts[3])
            width_norm = float(parts[4])
            height_norm = float(parts[5])

            abs_x = x_center_norm * frame_w
            abs_y = y_center_norm * frame_h
            abs_w = width_norm * frame_w
            abs_h = height_norm * frame_h

            x1 = int(abs_x - abs_w / 2)
            y1 = int(abs_y - abs_h / 2)
            x2 = int(abs_x + abs_w / 2)
            y2 = int(abs_y + abs_h / 2)

            gt_data[frame_id].append(Player(player_id=obj_id, bbox=[x1, y1, x2, y2]))

    print(f"Loaded GT data for {len(gt_data)} frames.")
    return gt_data


def load_ground_truth_absolute(gt_path: str) -> PlayersDetections:
    """
    Load ground truth with absolute coordinates (x, y, w, h)
    and convert to xyxy format.
    """
    gt_data: PlayersDetections = collections.defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt_data[frame_id].append(Player(player_id=obj_id, bbox=[int(x), int(y), int(x + w), int(y + h)]))
    return gt_data
