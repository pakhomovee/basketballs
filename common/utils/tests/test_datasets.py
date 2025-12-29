import cv2
import pytest
from common.utils.datasets import TeamTrackDataset
from common.utils.utils import get_sample_video_path

@pytest.fixture
def sample_frame():
    """Extracts a frame from the sample video."""
    video_path = get_sample_video_path()
    if video_path is None:
        pytest.skip("Video file not found.")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        pytest.fail(f"Could not read frame from video at {video_path}")
    return frame

def test_dataset_load():
    """Test if the dataset loads correctly."""
    dataset = TeamTrackDataset("../../common/data/train/sample_item")
    assert len(dataset) == 900
    assert len(dataset.annotations) == 900
    assert len(dataset.items) == 900
    assert len(dataset.video_metadata) == 1
    assert len(dataset.video_files) == 1
