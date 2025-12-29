import cv2
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

def get_sample_video_path():
    # for bazel test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "../../common/data/train/sample_item/img1.mp4")
    if os.path.exists(path):
        return path
    # for local test
    path = "common/data/train/sample_item/img1.mp4"
    if os.path.exists(path):
        return path

    return None

def play_video(video):
    try:
        while(True):
            ret, frame = video.read()
            if not ret:
                video.release()
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.axis('off')
            plt.title("Input Stream")
            plt.imshow(frame)
            plt.show()
            clear_output(wait=True)
    except KeyboardInterrupt:
        video.release()