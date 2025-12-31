import cv2
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

def get_dir(path):
    # for bazel test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp = os.path.join(script_dir, path)
    if os.path.exists(tmp):
        return tmp
    # for local test
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
