import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from tqdm import tqdm


def apply_H(pt_h, H):
    """Apply 3x3 homography to homogeneous point."""
    mapped = H @ pt_h
    if mapped[2] != 0:
        mapped = mapped / mapped[2]
    return mapped


def make_mask(shape, scoreboard=(0, 850, 1920, 0), people=None):
    h, w = shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    x1, y1, x2, y2 = scoreboard
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x1c < x2c and y1c < y2c:
        mask[y1c:y2c, x1c:x2c] = 0
    if people:
        for px, py, pw, ph in people:
            px2, py2 = px + pw, py + ph
            px1c, py1c, px2c, py2c = max(0, px), max(0, py), min(w, px2), min(h, py2)
            if px1c < px2c and py1c < py2c:
                mask[py1c:py2c, px1c:px2c] = 0
    return mask


def load_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    grays = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames, grays


def resize_for_detection(frame, max_side=640):
    h, w = frame.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame
    return resized, scale


def detect_people_batch(
    frames,
    model,
    device,
    transform,
    max_people=15,
    score_thresh=0.3,
    batch_size=32,
    max_side=640,
):
    detections = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            resized_batch = []
            scales = []
            for f in batch_frames:
                resized, scale = resize_for_detection(f, max_side=max_side)
                resized_batch.append(resized)
                scales.append(scale)

            images = [transform(f).to(device) for f in resized_batch]
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            for out, scale in zip(outputs, scales):
                boxes = out["boxes"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                people = []
                for b, label, s in zip(boxes, labels, scores):
                    if label != 1 or s < score_thresh:  # COCO label 1 = person
                        continue
                    x1, y1, x2, y2 = b.astype(int)
                    # rescale back to original frame coords
                    if scale != 1.0 and scale > 0:
                        inv = 1.0 / scale
                        x1 = int(x1 * inv)
                        y1 = int(y1 * inv)
                        x2 = int(x2 * inv)
                        y2 = int(y2 * inv)
                    people.append((x1, y1, x2 - x1, y2 - y1))
                people = sorted(people, key=lambda r: r[2] * r[3], reverse=True)
                detections.append(people[:max_people])
            torch.cuda.empty_cache()
    return detections


def compute_forward_homographies(frames, grays, scoreboard, people_detections, show_progress=False):
    n = len(frames)
    forward_H = [None] * (n - 1)

    if show_progress:
        cv2.namedWindow("progress", cv2.WINDOW_NORMAL)

    for i in range(n - 1):
        people_rects = people_detections[i]
        mask = make_mask(grays[i].shape, scoreboard=scoreboard, people=people_rects)

        prev_pts = cv2.goodFeaturesToTrack(
            grays[i],
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7,
            useHarrisDetector=False,
            k=0.04,
            mask=mask,
        )
        if prev_pts is None:
            continue
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(grays[i], grays[i + 1], prev_pts, None)
        if curr_pts is None or status is None:
            continue
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        if len(good_prev) < 4:
            continue
        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
        if H is not None and inliers is not None:
            forward_H[i] = H

        if show_progress:
            vis = frames[i].copy()
            for px, py, pw, ph in people_rects:
                cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            cv2.imshow("progress", vis)
            cv2.waitKey(1)

    if show_progress:
        cv2.destroyWindow("progress")
    return forward_H


def accumulate_ref_transforms(forward_H, ref_idx):
    n = len(forward_H) + 1
    ref_to = [None] * n
    ref_to[ref_idx] = np.eye(3, dtype=np.float32)

    # forward (ref -> higher index)
    for i in range(ref_idx + 1, n):
        H_prev = forward_H[i - 1]
        if H_prev is None:
            ref_to[i] = ref_to[i - 1]
        else:
            ref_to[i] = ref_to[i - 1] @ H_prev

    # backward (ref -> lower index)
    for i in range(ref_idx - 1, -1, -1):
        H_i = forward_H[i]
        if H_i is None:
            ref_to[i] = ref_to[i + 1]
        else:
            try:
                H_inv = np.linalg.inv(H_i)
            except np.linalg.LinAlgError:
                H_inv = np.eye(3, dtype=np.float32)
            ref_to[i] = ref_to[i + 1] @ H_inv

    return ref_to


def accumulate_from_zero(forward_H):
    """Compute cumulative transforms H_0_to_i for all i."""
    n = len(forward_H) + 1
    H0_to = [None] * n
    H0_to[0] = np.eye(3, dtype=np.float32)
    for i in range(1, n):
        if H0_to[i - 1] is None or forward_H[i - 1] is None:
            H0_to[i] = None
        else:
            H0_to[i] = H0_to[i - 1] @ forward_H[i - 1]
    return H0_to


def safe_inv(H):
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None


def choose_ref_by_center_alignment(forward_H, frame_shape):
    """Pick reference frame minimizing max distance of projected centers."""
    h, w = frame_shape
    center_ref = np.array([w / 2.0, h / 2.0, 1.0], dtype=np.float32)
    H0_to = accumulate_from_zero(forward_H)
    n = len(H0_to)
    best_ref = 0
    best_score = np.inf

    # Precompute inverses
    inv_H0_to = [safe_inv(H) if H is not None else None for H in H0_to]

    for r in tqdm(range(n)):
        if H0_to[r] is None:
            continue
        max_dist = -np.inf
        for i in range(n):
            if H0_to[i] is None or inv_H0_to[i] is None:
                continue
            H_i_to_r = H0_to[r] @ inv_H0_to[i]
            p = apply_H(center_ref, H_i_to_r)
            d = np.linalg.norm(p[:2] - center_ref[:2])
            if d > max_dist:
                max_dist = d
        if max_dist < best_score:
            best_score = max_dist
            best_ref = r

    return best_ref


def build_panorama(frames, ref_to, ref_idx):
    h, w = frames[ref_idx].shape[:2]
    pano_canvas = np.zeros((h * 2, w * 4, 3), dtype=np.uint8)
    T_translate = np.array([[1, 0, w * 1.5], [0, 1, h * 0.5], [0, 0, 1]], dtype=np.float32)
    for i, frame in enumerate(frames):
        H_ref_to_i = ref_to[i]
        if H_ref_to_i is None:
            continue
        try:
            H_i_to_ref = np.linalg.inv(H_ref_to_i)
        except np.linalg.LinAlgError:
            continue
        H_warp = T_translate @ H_i_to_ref
        cv2.warpPerspective(
            frame,
            H_warp,
            (pano_canvas.shape[1], pano_canvas.shape[0]),
            dst=pano_canvas,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
    return pano_canvas, T_translate


def main():
    video_path = Path(__file__).parent / "dataset" / "segment.mp4"
    frames, grays = load_frames(video_path)
    if not frames:
        print("No frames loaded.")
        return

    scoreboard = (450, 850, 1450, 1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT").to(device)
    detector.eval()
    transform = T.Compose([T.ToTensor()])

    people_detections = detect_people_batch(frames, detector, device, transform, batch_size=4, max_side=640)

    forward_H = compute_forward_homographies(frames, grays, scoreboard, people_detections, show_progress=True)
    ref_idx = choose_ref_by_center_alignment(forward_H, grays[0].shape)
    print(f"Reference frame index (center alignment): {ref_idx}")

    ref_to = accumulate_ref_transforms(forward_H, ref_idx)
    pano_canvas, T_translate = build_panorama(frames, ref_to, ref_idx)

    cv2.namedWindow("features", cv2.WINDOW_NORMAL)
    cv2.namedWindow("panorama", cv2.WINDOW_NORMAL)

    tracked_points_ref = []  # points stored in reference frame coordinates (homogeneous)
    latest_vis = [None]
    idx_state = {"idx": ref_idx}

    def redraw(vis):
        latest_vis[0] = vis
        disp = vis.copy()
        curr_idx = idx_state["idx"]
        H_ref_to_curr = ref_to[curr_idx]
        if H_ref_to_curr is None:
            H_ref_to_curr = np.eye(3, dtype=np.float32)
        for p_ref in tracked_points_ref:
            p_curr = apply_H(p_ref, H_ref_to_curr)
            x, y = int(p_curr[0]), int(p_curr[1])
            cv2.circle(disp, (x, y), 6, (255, 0, 255), -1)
        cv2.imshow("features", disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and latest_vis[0] is not None:
            curr_idx = idx_state["idx"]
            H_ref_to_curr = ref_to[curr_idx]
            if H_ref_to_curr is None:
                H_ref_to_curr = np.eye(3, dtype=np.float32)
            try:
                H_curr_to_ref = np.linalg.inv(H_ref_to_curr)
            except np.linalg.LinAlgError:
                H_curr_to_ref = np.eye(3, dtype=np.float32)
            pt_ref = apply_H(np.array([float(x), float(y), 1.0], dtype=np.float32), H_curr_to_ref)
            tracked_points_ref.append(pt_ref)
            redraw(latest_vis[0])

    cv2.setMouseCallback("features", on_mouse)

    while True:
        idx = idx_state["idx"] % len(frames)
        frame = frames[idx].copy()
        redraw(frame)
        cv2.imwrite("panorama.png", pano_canvas)
        cv2.imshow("panorama", pano_canvas)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            idx_state["idx"] = (idx_state["idx"] + 1) % len(frames)
        if key == ord("a"):
            idx_state["idx"] = (idx_state["idx"] - 1) % len(frames)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
