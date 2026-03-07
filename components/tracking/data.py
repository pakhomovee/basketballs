"""
Data loading utilities for tracking.

- get_measurements: filter detections for a frame
- load_detections_csv: load tracking-format CSV into detection dicts
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_measurements(frame_id, data):
    """Extract measurements for a given frame from detection data."""
    out = []
    for d in data:
        if d["frame_id"] != frame_id or d.get("class", "player") != "player":
            continue
        m = {
            "field_coords": np.array(d["field_coords"]),
            "bbox": np.array(d["bbox"]),
        }
        if "_player_ref" in d:
            m["_player_ref"] = d["_player_ref"]
        out.append(m)
    return out


def load_detections_csv(csv_path):
    """Load tracking-format CSV into a list of detection dicts."""
    df = pd.read_csv(csv_path, header=None)
    df.columns = [
        "frame_id",
        "class_id",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_x_max",
        "bbox_y_max",
        "skip1",
        "skip2",
        "skip3",
        "skip4",
        "field_x",
        "field_y",
    ]
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "frame_id": int(r["frame_id"]),
                "detection_id": f"d{int(r['class_id'])}",
                "bbox": [
                    int(r["bbox_x_min"]),
                    int(r["bbox_y_min"]),
                    int(r["bbox_x_max"]),
                    int(r["bbox_y_max"]),
                ],
                "field_coords": [float(r["field_x"]), float(r["field_y"])],
                "class": "player",
            }
        )
    return out
