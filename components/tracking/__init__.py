"""
Basketball player tracking.

Public API
----------
- PlayerTracker : online Kalman + IoU + appearance tracker
- FlowTracker   : offline min-cost-max-flow tracker (globally optimal)
- Track, TrackState : track representation
- load_detections_csv, get_measurements : data loading
- evaluate_tracking, run_benchmark, match_objects : evaluation
- bbox_iou : distance utility
"""

from .data import get_measurements, load_detections_csv
from common.distances import bbox_iou
from .evaluation import evaluate_tracking, match_objects, run_benchmark
from .flow_tracker import FlowTracker

__all__ = [
    "FlowTracker",
    "load_detections_csv",
    "get_measurements",
    "evaluate_tracking",
    "run_benchmark",
    "match_objects",
    "bbox_iou",
]
