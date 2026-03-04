import cv2
import numpy as np

def make_side_by_side_video(
    top_video_path: str,
    bottom_video_path: str,
    output_path: str,
) -> None:
    """
    Создать видео, где два входных видео идут одновременно сверху и снизу.

    top_video_path    — путь к первому (верхнему) видео.
    bottom_video_path — путь ко второму (нижнему) видео.
    output_path       — путь к результирующему видео (например, 'out.mp4').
    """
    print("Writing side by side (top / bottom)")

    cap_top = cv2.VideoCapture(top_video_path)
    cap_bottom = cv2.VideoCapture(bottom_video_path)

    if not cap_top.isOpened():
        raise RuntimeError(f"Cannot open top video: {top_video_path}")
    if not cap_bottom.isOpened():
        raise RuntimeError(f"Cannot open bottom video: {bottom_video_path}")

    fps = cap_top.get(cv2.CAP_PROP_FPS) or 25.0
    top_w = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
    top_h = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bottom_w = int(cap_bottom.get(cv2.CAP_PROP_FRAME_WIDTH))
    bottom_h = int(cap_bottom.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Общая ширина — по верхнему видео; нижнее подгоняем по ширине
    target_w = top_w
    scale_bottom = target_w / float(bottom_w) if bottom_w > 0 else 1.0
    bottom_h_resized = int(bottom_h * scale_bottom)
    target_bottom_w = target_w
    target_bottom_h = bottom_h_resized

    out_w = target_w
    out_h = top_h + target_bottom_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    try:
        while True:
            ret_t, frame_top = cap_top.read()
            ret_b, frame_bottom = cap_bottom.read()
            if not ret_t or not ret_b:
                break

            if frame_top.shape[:2] != (top_h, top_w):
                frame_top = cv2.resize(frame_top, (target_w, top_h))
            frame_bottom = cv2.resize(frame_bottom, (target_bottom_w, target_bottom_h))

            # Сверху — первое видео, снизу — второе
            combined = np.concatenate([frame_top, frame_bottom], axis=0)
            writer.write(combined)
    finally:
        cap_top.release()
        cap_bottom.release()
        writer.release()
        print(f"Side by side video saved to {output_path}")
