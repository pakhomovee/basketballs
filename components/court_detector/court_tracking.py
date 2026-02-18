import cv2
import numpy as np
from copy import deepcopy
import math
from geometry import (
    GeometricLine,
    intersect_lines,
    check_point_inside_frame,
    extend_segment_by_frame,
    crop_segment_by_frame,
    dist,
)
from court_constants import COURT_POINTS, COURT_LINES


def apply_homography_to_point(point, H):
    x, y = point
    p = np.array([x, y, 1])
    transformed = np.dot(H, p)
    res = transformed[:2] / transformed[2]
    return (res[0], res[1])


def sample_tracking_points(H, frame_size, extend_lines=True, samples_per_line=100):
    """
    This function samples integer points on court lines.
    If extend_lines is True (default value), the court lines are extended to the whole frame.
    """
    points = []
    for i, point in enumerate(COURT_POINTS):
        new_point = apply_homography_to_point(point, H)
        points.append(new_point if check_point_inside_frame(new_point, frame_size) else None)
    lines = []
    for i, j in COURT_LINES:
        if points[i] is None or points[j] is None:
            continue
        line = (points[i], points[j])
        line = crop_segment_by_frame(line, frame_size)
        if line is None:
            continue
        if extend_lines:
            line = extend_segment_by_frame(line, frame_size)
        lines.append(line)
    sampled_points = []
    for line in lines:
        if line is None:
            continue
        x1, y1 = line[0]
        x2, y2 = line[1]
        for i in range(samples_per_line):
            alpha = (i + 1) / (samples_per_line + 1)
            px = x1 * alpha + x2 * (1 - alpha)
            py = y1 * alpha + y2 * (1 - alpha)
            px = int(px)
            py = int(py)
            if check_point_inside_frame((px, py), frame_size):
                sampled_points.append((px, py))
    return sampled_points


def recalculate_homography_from_detected_lines(H, court_lines, frame_lines, eps=1e-2):
    A = []
    # First we norm all three dimensions to get more well-defined system
    court_lines_norm = np.array([1, 1, 1])
    frame_lines_norm = np.array([1, 1, 1])
    for court_line, frame_line in zip(court_lines, frame_lines):
        court_line_eq = np.array(court_line.get_equation())
        frame_line_eq = np.array(frame_line.get_equation())
        court_lines_norm = court_lines_norm + court_line_eq**2
        frame_lines_norm = frame_lines_norm + frame_line_eq**2
    court_lines_norm = np.sqrt(court_lines_norm)
    frame_lines_norm = np.sqrt(frame_lines_norm)
    for court_line, frame_line in zip(court_lines, frame_lines):
        court_line_eq = np.array(court_line.get_equation())
        frame_line_eq = np.array(frame_line.get_equation())

        court_line_eq /= court_lines_norm
        frame_line_eq /= frame_lines_norm

        idx_pairs = []
        if abs(court_line_eq[0]) > abs(court_line_eq[1]):
            idx_pairs.append((0, 1))
            idx_pairs.append((0, 2))
        else:
            idx_pairs.append((1, 0))
            idx_pairs.append((1, 2))
        for i, j in idx_pairs:
            scalar = np.zeros((3, 3))
            a = court_line_eq[i]
            b = court_line_eq[j]
            scalar[i] = frame_line_eq * b
            scalar[j] = -frame_line_eq * a
            A.append(scalar.flatten())
    if len(A) == 0:
        return H
    A = np.array(A)
    h = (np.diag(1 / court_lines_norm) @ H.T @ np.diag(frame_lines_norm)).flatten()
    U, S, Vh = np.linalg.svd(A)
    S.resize(9)
    g = Vh @ h
    g = g / np.linalg.norm(g)
    mask = np.zeros((9))
    for i in range(9):
        if abs(S[i] - S[-1]) < eps:
            mask[i] = 1
    sol = mask * g
    sol = sol / np.linalg.norm(sol)
    print(np.dot(sol, g))
    print(S)
    sol = Vh.T @ sol

    sol = sol.reshape((3, 3))
    sol = np.diag(court_lines_norm) @ sol @ np.diag(1 / frame_lines_norm)
    return sol.T


def recalculate_homography(H, old_frame, frame, min_length=200, max_lines=6, threshold=30):
    h, w, _ = frame.shape
    old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    old_points = sample_tracking_points(H, (w, h))
    old_points = np.array(old_points, dtype=np.float32).reshape((-1, 1, 2))

    if len(old_points) < 20:
        return reference_points

    lk_params = dict(
        winSize=(30, 30),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    new_points, st, err = cv2.calcOpticalFlowPyrLK(
        old_frame_gray, frame_gray, old_points, None, **lk_params
    )

    st = st.flatten().astype(dtype=np.bool)
    old_points = old_points[st]
    new_points = new_points[st]

    H_delta, err = cv2.findHomography(old_points, new_points)

    if H_delta is None:
        H_lk = H
    else:
        H_lk = H_delta @ H

    edges = cv2.Canny(frame, 100, 200)

    court_lines = []
    frame_lines = []

    for i, j in COURT_LINES:
        pi = apply_homography_to_point(COURT_POINTS[i], H_lk)
        pj = apply_homography_to_point(COURT_POINTS[j], H_lk)
        line = crop_segment_by_frame((pi, pj), (w, h))
        if line is None:
            continue
        pi, pj = line
        if dist(pi, pj) < min_length:
            continue

        mask = np.zeros((h, w), dtype="uint8")
        cv2.line(mask, pi, pj, 255, threshold)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        hough_lines = cv2.HoughLines(masked_edges, 1, np.pi / 180, 20, None, 0, 0)
        if hough_lines is None:
            continue
        best_line = None
        best_line_dist_sum = 2 * threshold + 10
        for line_idx in range(0, min(len(hough_lines), max_lines)):
            rho = hough_lines[line_idx][0][0]
            theta = hough_lines[line_idx][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            c = -rho
            hough_line = GeometricLine(a, b, c)
            if hough_line.dist(pi) > threshold or hough_line.dist(pj) > threshold:
                continue
            dist_sum = hough_line.dist(pi) + hough_line.dist(pj)
            if dist_sum > best_line_dist_sum:
                continue
            best_line_dist_sum = dist_sum
            best_line = hough_line
        if best_line is not None:
            court_lines.append(GeometricLine.from_points(COURT_POINTS[i], COURT_POINTS[j]))
            frame_lines.append(best_line)

    H_new = recalculate_homography_from_detected_lines(H_lk, court_lines, frame_lines)

    # court_lines = get_court_lines(reference_points)

    # for i, court_line in enumerate(court_lines):
    #     if court_line is None:
    #         continue
    #     p1, p2 = court_line
    #     mask = np.zeros((h, w), dtype="uint8")
    #     cv2.line(mask, p1, p2, 255, 40)
    #     masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    #     lines = cv2.HoughLines(masked_edges, 1, np.pi / 180, 20, None, 0, 0)
    #     if lines is not None:
    #         for j in range(0, min(len(lines), 3)):
    #             rho = lines[j][0][0]
    #             theta = lines[j][0][1]
    #             a = math.cos(theta)
    #             b = math.sin(theta)
    #             x0 = a * rho
    #             y0 = b * rho
    #             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #             geometric_line = GeometricLine(pt1, pt2)
    #             if geometric_line.dist(p1) <= 40 and geometric_line.dist(p2) <= 40:
    #                 court_lines[i] = (pt1, pt2)
    #                 break

    # for i in range(len(new_points)):
    #     my_lines = []
    #     for j in range(len(court_lines)):
    #         if court_lines[j] is None:
    #             continue
    #         if i in points_on_lines[j]:
    #             my_lines.append(GeometricLine(*court_lines[j]))
    #     if len(my_lines) >= 2:
    #         x, y = intersect_lines(my_lines[0], my_lines[1])
    #         new_points[i] = (int(x), int(y))
    #         print(i,new_points[i])
    #     elif len(my_lines) == 1 and new_points[i] is not None:
    #         x, y = my_lines[0].project(new_points[i])
    #         new_points[i] = (int(x), int(y))
    #         print(i,new_points[i])

    return H_new


cap = cv2.VideoCapture("dataset/segment.mp4")

reference_points = [None for i in range(len(COURT_POINTS))]
cur_reference_point = 0
H = None


def try_calculate_homography():
    global H, reference_points
    points = []
    new_points = []
    for i, p in enumerate(COURT_POINTS):
        if reference_points[i] is None:
            continue
        points.append(p)
        new_points.append(reference_points[i])
    points = np.array(points).reshape((-1, 1, 2))
    new_points = np.array(new_points).reshape((-1, 1, 2))
    try:
        H_new, err = cv2.findHomography(points, new_points)
    except cv2.error:
        return
    H = H_new


def on_mouse(event, x, y, flags, param):
    global cur_reference_point, reference_points
    if (
        event == cv2.EVENT_LBUTTONDOWN
        or event == cv2.EVENT_RBUTTONDOWN
        or event == cv2.EVENT_MBUTTONDOWN
    ):
        if event == cv2.EVENT_LBUTTONDOWN:
            reference_points[cur_reference_point] = (x, y)
            try_calculate_homography()
            show_frame(frame, reference_points)
        cur_reference_point = (cur_reference_point + 1) % len(COURT_POINTS)


cv2.namedWindow("match")
cv2.setMouseCallback("match", on_mouse)


def show_frame(frame, reference_points):
    frame = deepcopy(frame)
    edges = cv2.Canny(frame, 100, 200)
    h, w, _ = frame.shape
    for p in reference_points:
        if p is None:
            continue
        x, y = p
        cv2.circle(frame, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)

    if H is not None:
        for p in COURT_POINTS:
            p_new = apply_homography_to_point(p, H)
            if check_point_inside_frame(p_new, (w, h)):
                x, y = p_new
                cv2.circle(
                    frame, (int(round(x)), int(round(y))), radius=4, color=(0, 255, 0), thickness=-1
                )

    # sampled_points = sample_tracking_points(reference_points, (w, h))
    # for x, y in sampled_points:
    #     cv2.circle(frame, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
    # for court_line in get_court_lines(reference_points):
    #     if court_line is None:
    #         continue
    #     p1, p2 = court_line
    #     mask = np.zeros((h, w), dtype="uint8")
    #     cv2.line(mask, p1, p2, 255, 40)
    #     masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    #     lines = cv2.HoughLines(masked_edges, 1, np.pi / 180, 20, None, 0, 0)
    #     if lines is not None:
    #         for i in range(0, min(len(lines), 3)):
    #             rho = lines[i][0][0]
    #             theta = lines[i][0][1]
    #             a = math.cos(theta)
    #             b = math.sin(theta)
    #             x0 = a * rho
    #             y0 = b * rho
    #             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #             # cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    cv2.imwrite("frame.png", frame)
    cv2.imshow("match", frame)
    cv2.imshow("edges", edges)


old_frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cur_reference_point = 0
    reference_points = [None for i in range(len(COURT_POINTS))]
    if old_frame is not None and H is not None:
        H = recalculate_homography(H, old_frame, frame)

    old_frame = deepcopy(frame)
    show_frame(frame, reference_points)

    shoud_exit = False
    while True:
        key = cv2.waitKey(-1)
        if key & 0xFF == ord("q"):
            shoud_exit = True
            break
        if key & 0xFF == ord(" "):
            break
    if shoud_exit:
        break

cap.release()
cv2.destroyAllWindows()
