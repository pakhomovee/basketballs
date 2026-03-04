"""Bounding box visualization utilities for detections."""

import cv2


def visualize_detection(frame, detections):
    """Draw bounding boxes on frame for each detection (x1, y1, x2, y2, score)."""
    for detection in detections:
        x1, y1, x2, y2, score = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame


def visualize_detection_with_confidence(frame, detections):
    """Draw bounding boxes with confidence scores on frame."""
    for detection in detections:
        x1, y1, x2, y2, score = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame
