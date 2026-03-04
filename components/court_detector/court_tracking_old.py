import cv2
import numpy as np
from copy import deepcopy
import math
from geometry import GeometricLine, intersect_lines

points_on_lines = [
    [0, 6],
    [1, 2, 8, 7],
    [4, 3, 9, 10],
    [5, 11],
    [0, 1, 4, 5],
    [2, 3],
    [8, 9],
    [6, 7, 10, 11],
]


def get_court_lines(reference_points):
    """
    This function calculates 8 lines from 12 reference points.
    Line is represented as a pair of points.
    Since some points could be None, some lines also could be None.
    """
    lines = [None for i in range(8)]
    for i in range(len(lines)):
        points = points_on_lines[i]
        lft = 0
        rght = len(points) - 1
        while lft < rght:
            l_bad = reference_points[points[lft]] is None
            r_bad = reference_points[points[rght]] is None
            if (not l_bad) and (not r_bad):
                break
            if l_bad:
                lft += 1
            if r_bad:
                rght -= 1
        if lft < rght:
            lines[i] = (reference_points[points[lft]], reference_points[points[rght]])
    return lines


def check_point_inside_frame(point, frame_size):
    """
    This function checks if a point is inside the frame.
    """
    x, y = point
    w, h = frame_size
    return 0 <= x and x < w and 0 <= y and y < h


def extend_line(line, frame_size, infinity=1e9):
    """
    This function takes a line (represented by a pair of points) and extends it to the whole frame.
    The line should lie inside the frame!
    """
    w, h = frame_size
    x1, y1 = line[0]
    x2, y2 = line[1]

    if (not check_point_inside_frame((x1, y1), frame_size)) or (not check_point_inside_frame((x2, y2), frame_size)):
        raise Exception("Line doesn't lie inside the frame!")

    min_alpha, max_alpha = -infinity, infinity

    def add_constraint(coord1, coord2, min_coord, max_coord):
        nonlocal min_alpha, max_alpha
        if abs(coord1 - coord2) > 0:
            k, b = coord2 - coord1, coord1
            if k < 0:
                k, b = -k, -b
                min_coord, max_coord = -max_coord, -min_coord
            min_alpha = max(min_alpha, (min_coord - b) / k)
            max_alpha = min(max_alpha, (max_coord - b) / k)

    add_constraint(x1, x2, 0, w - 1)
    add_constraint(y1, y2, 0, h - 1)

    assert min_alpha <= max_alpha

    nx1, ny1 = x1 + (x2 - x1) * min_alpha, y1 + (y2 - y1) * min_alpha
    nx2, ny2 = x1 + (x2 - x1) * max_alpha, y1 + (y2 - y1) * max_alpha

    nx1, ny1, nx2, ny2 = int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))

    assert check_point_inside_frame((nx1, ny1), frame_size) and check_point_inside_frame((nx2, ny2), frame_size)

    return ((nx1, ny1), (nx2, ny2))


def sample_tracking_points(reference_points, frame_size, extend_lines=True, samples_per_line=100):
    """
    This function samples integer points on court lines.
    If extend_lines is True (default value), the court lines are extended to the whole frame.
    """
    lines = get_court_lines(reference_points)
    sampled_points = []
    for line in lines:
        if line is None:
            continue
        if extend_lines:
            line = extend_line(line, frame_size)
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


def recalculate_reference_points(reference_points, old_frame, frame):
    h, w, _ = frame.shape
    old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    old_points = sample_tracking_points(reference_points, (w, h))
    old_points = np.array(old_points, dtype=np.float32).reshape((-1, 1, 2))

    if len(old_points) < 20:
        return reference_points

    new_points, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, frame_gray, old_points, None, **lk_params)
    st = st.flatten().astype(dtype=np.bool)
    old_points = old_points[st]
    new_points = new_points[st]

    H, err = cv2.findHomography(old_points, new_points)

    new_points = deepcopy(reference_points)
    for i in range(len(reference_points)):
        p = reference_points[i]
        if p is not None:
            p = apply_homography_to_point(p, H)
            x, y = p
            x, y = int(x), int(y)
            p = (x, y)
            if check_point_inside_frame(p, (w, h)):
                new_points[i] = p
            else:
                new_points[i] = None

    edges = cv2.Canny(frame, 100, 200)

    court_lines = get_court_lines(reference_points)

    for i, court_line in enumerate(court_lines):
        if court_line is None:
            continue
        p1, p2 = court_line
        mask = np.zeros((h, w), dtype="uint8")
        cv2.line(mask, p1, p2, 255, 40)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        lines = cv2.HoughLines(masked_edges, 1, np.pi / 180, 20, None, 0, 0)
        if lines is not None:
            for j in range(0, min(len(lines), 3)):
                rho = lines[j][0][0]
                theta = lines[j][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                geometric_line = GeometricLine.from_points(pt1, pt2)
                if geometric_line.dist(p1) <= 40 and geometric_line.dist(p2) <= 40:
                    court_lines[i] = (pt1, pt2)
                    break

    for i in range(len(new_points)):
        my_lines = []
        for j in range(len(court_lines)):
            if court_lines[j] is None:
                continue
            if i in points_on_lines[j]:
                my_lines.append(GeometricLine.from_points(*court_lines[j]))
        if len(my_lines) >= 2:
            x, y = intersect_lines(my_lines[0], my_lines[1])
            new_points[i] = (int(x), int(y))
            print(i, new_points[i])
        elif len(my_lines) == 1 and new_points[i] is not None:
            x, y = my_lines[0].project(new_points[i])
            new_points[i] = (int(x), int(y))
            print(i, new_points[i])

    return new_points


def apply_homography_to_point(point, H):
    x, y = point
    p = np.array([x, y, 1])
    transformed = np.dot(H, p)
    res = transformed[:2] / transformed[2]
    return (res[0], res[1])


cap = cv2.VideoCapture("dataset/segment.mp4")

reference_points = [None for i in range(12)]
cur_reference_point = 0


def on_mouse(event, x, y, flags, param):
    global user_points, cur_reference_point
    if event == cv2.EVENT_LBUTTONDOWN:
        reference_points[cur_reference_point] = (x, y)
        cur_reference_point = (cur_reference_point + 1) % 12
        show_frame(frame, reference_points)


cv2.namedWindow("match")
cv2.setMouseCallback("match", on_mouse)


def show_frame(frame, reference_points):
    frame = deepcopy(frame)
    edges = cv2.Canny(frame, 100, 200)
    h, w, _ = frame.shape
    # sampled_points = sample_tracking_points(reference_points, (w, h))
    # for x, y in sampled_points:
    #     cv2.circle(frame, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
    for court_line in get_court_lines(reference_points):
        if court_line is None:
            continue
        p1, p2 = court_line
        mask = np.zeros((h, w), dtype="uint8")
        cv2.line(mask, p1, p2, 255, 40)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        lines = cv2.HoughLines(masked_edges, 1, np.pi / 180, 20, None, 0, 0)
        if lines is not None:
            for i in range(0, min(len(lines), 3)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
                pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    for p in reference_points:
        if p is None:
            continue
        x, y = p
        cv2.circle(frame, (int(x), int(y)), radius=7, color=(0, 255, 0), thickness=-1)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    cv2.imwrite("frame.png", frame)
    cv2.imshow("match", frame)
    cv2.imshow("edges", edges)


# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(30, 30),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

old_frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    points = []
    h, w, _ = frame.shape
    if old_frame is not None:
        reference_points = recalculate_reference_points(reference_points, old_frame, frame)

    old_frame = frame[:]
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
