import cv2
import pytest
import os
import sys
import torch
import numpy as np
from components.detector.detector_model import detect_persons_torch, model
from visualization.visualize import visualize_detection

# sample video
def get_video_path():
    # for bazel test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "data", "Q2_side_540-570.mp4")
    if os.path.exists(path):
        return path
    # for local test
    path = "components/detector/tests/data/Q2_side_540-570.mp4"
    if os.path.exists(path):
        return path
    
    return None

@pytest.fixture
def sample_frame():
    """Extracts a frame from the sample video."""
    video_path = get_video_path()
    if video_path is None:
        pytest.skip(f"Video file not found. Checked relative to script and CWD.")
        
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        pytest.fail(f"Could not read frame from video at {video_path}")
    return frame

def test_model_load():
    """Test if the model loads correctly."""
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_detection_output_format(sample_frame):
    """Test if the detection output has the correct format."""
    persons = detect_persons_torch(sample_frame)
    assert isinstance(persons, list)
    if len(persons) > 0:
        for person in persons:
            assert len(person) == 5
            x1, y1, x2, y2, score = person
            assert isinstance(x1, int)
            assert isinstance(y1, int)
            assert isinstance(x2, int)
            assert isinstance(y2, int)
            assert isinstance(score, (float, np.float32, np.float64))
            assert x1 < x2
            assert y1 < y2
            assert score >= 0.0 and score <= 1.0
    
    # For manual checking
    print(f"\nNumber of detections: {len(persons)}")
    if len(persons) == 0:
        print("WARNING: No persons detected!")
        
    vis_frame = visualize_detection(sample_frame.copy(), persons)
    output_path = "test_detection_output.jpg"
    cv2.imwrite(output_path, vis_frame)
    print(f"\nSaved visualization to: {os.path.abspath(output_path)}")

def test_detection_consistency(sample_frame):
    """Test if running detection twice on the same frame yields same results."""
    persons1 = detect_persons_torch(sample_frame)
    persons2 = detect_persons_torch(sample_frame)
    assert len(persons1) == len(persons2)
    if len(persons1) > 0:
        assert persons1[0] == persons2[0]

def test_empty_frame():
    """Test behavior with a black frame."""
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    persons = detect_persons_torch(black_frame)
    assert isinstance(persons, list)

if __name__ == "__main__":
    # -s allows stdout/stderr to be seen
    sys.exit(pytest.main(["-v", "-s", __file__]))
