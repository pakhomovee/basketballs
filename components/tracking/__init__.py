"""Basketball player tracking.

Public API
----------
- FlowTracker        : offline min-cost-max-flow tracker
- evaluate           : compute MOTA/HOTA/IDF1 for one sequence
- run_benchmark      : benchmark runner over annotated dataset
- load_yolo_mot      : load YOLO MOT annotation format
- load_detections_csv, get_measurements : legacy CSV data loading
- bbox_iou           : distance utility
"""

from .data import get_measurements, load_detections_csv
from common.distances import bbox_iou
from .evaluation import evaluate, load_yolo_mot, match_frame, remap_pred_ids
from .benchmark import run_benchmark
from .flow_tracker import FlowTracker
from .hungarian_tracker import HungarianTracker
from .stitching import stitch_tracklets

__all__ = [
    "FlowTracker",
    "HungarianTracker",
    "stitch_tracklets",
    "evaluate",
    "load_yolo_mot",
    "match_frame",
    "run_benchmark",
    "load_detections_csv",
    "get_measurements",
    "bbox_iou",
]
