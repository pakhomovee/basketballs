from math import sqrt

EPSILON = 1e-6
INFINITY = 1e6


class GeometricLine:
    def __init__(self, a, b, c):

        length = sqrt(a**2 + b**2)
        a /= length
        b /= length
        c /= length
        self.a, self.b, self.c = a, b, c

    @classmethod
    def from_points(cls, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = -(a * x1 + b * y1)
        return cls(a, b, c)

    def dist(self, p):
        x, y = p
        return abs(self.a * x + self.b * y + self.c)

    def project(self, p):
        x, y = p
        d = self.a * x + self.b * y + self.c
        dx, dy = -d * self.a, -d * self.b
        return (x + dx, y + dy)

    def get_equation(self):
        """Returns (a, b, c) for the line a * x + b * y + c = 0"""
        return (self.a, self.b, self.c)


def intersect_lines(line1: GeometricLine, line2: GeometricLine):
    d = line1.b * line2.a - line2.b * line1.a
    if d == 0:
        return None
    dx = line2.b * line1.c - line2.c * line1.b
    dy = line2.c * line1.a - line1.c * line2.a
    return (dx / d, dy / d)


def check_point_inside_frame(point, frame_size):
    """
    This function checks if a point is inside the frame.
    """
    x, y = point
    w, h = frame_size
    return -EPSILON < x and x < (w - 1) + EPSILON and -EPSILON < y and y < (h - 1) + EPSILON


def _extend_crop_segment(segment, frame_size, min_alpha, max_alpha):
    """
    This is internal function.
    It takes segment, frame, and magic parameters min_alpha and max_alpha,
    and returns extended or croped segment (depending on magic parameters).
    """
    w, h = frame_size
    x1, y1 = segment[0]
    x2, y2 = segment[1]

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

    if min_alpha + EPSILON >= max_alpha:
        return None

    nx1, ny1 = x1 + (x2 - x1) * min_alpha, y1 + (y2 - y1) * min_alpha
    nx2, ny2 = x1 + (x2 - x1) * max_alpha, y1 + (y2 - y1) * max_alpha

    nx1, ny1, nx2, ny2 = int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))

    assert check_point_inside_frame((nx1, ny1), frame_size) and check_point_inside_frame((nx2, ny2), frame_size)

    return ((nx1, ny1), (nx2, ny2))


def extend_segment_by_frame(segment, frame_size):
    return _extend_crop_segment(segment, frame_size, -INFINITY, INFINITY)


def crop_segment_by_frame(segment, frame_size):
    return _extend_crop_segment(segment, frame_size, 0, 1)


def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
