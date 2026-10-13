"""
cra_analysis.py
------------------
Computes the crease recovery angle (the angle between the two arms of the
folded specimen after recovery) from a segmented image or from 3 clicked points.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class AngleResult:
    angle: float          # recovery angle in degrees (0-180)
    vertex: Point         # crease (fold) point
    end1: Point           # tip of arm 1
    end2: Point           # tip of arm 2
    method: str = "auto"


# ----------------------------------------------------------------------------
# Manual (3-point) method
# ----------------------------------------------------------------------------
def angle_from_points(vertex: Point, p1: Point, p2: Point) -> float:
    """Angle (degrees) at `vertex` between rays vertex->p1 and vertex->p2."""
    v1 = np.array(p1, dtype=float) - np.array(vertex, dtype=float)
    v2 = np.array(p2, dtype=float) - np.array(vertex, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Points must be different from the vertex.")
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def manual_result(vertex: Point, p1: Point, p2: Point) -> AngleResult:
    return AngleResult(angle_from_points(vertex, p1, p2), vertex, p1, p2, "manual")


# ----------------------------------------------------------------------------
# Automatic method
# ----------------------------------------------------------------------------
def _line_intersection(l1, l2) -> Optional[Point]:
    """Each line is (vx, vy, x0, y0) as returned by cv2.fitLine."""
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    det = vx1 * (-vy2) - (-vx2) * vy1
    if abs(det) < 1e-9:
        return None
    t = ((x2 - x1) * (-vy2) - (-vx2) * (y2 - y1)) / det
    return float(x1 + t * vx1), float(y1 + t * vy1)


def detect_angle_auto(
    mask: np.ndarray,
    min_line_frac: float = 0.12,
    hough_threshold: int = 40,
    max_line_gap: int = 20,
    min_separation_deg: float = 8.0,
) -> Optional[AngleResult]:
    """
    Find the two arms of the folded specimen in a binary mask.

    Steps
    1. Canny edges of the mask.
    2. Probabilistic Hough transform -> straight segments.
    3. Cluster segment orientations into two groups (the two arms).
    4. Fit one line per group (cv2.fitLine), intersect -> crease vertex.
    5. Angle between the arms measured from the vertex toward each arm's tip.

    Returns None if two distinct arms cannot be found.
    """
    h, w = mask.shape[:2]
    edges = cv2.Canny(mask, 50, 150)
    min_len = int(min_line_frac * max(h, w))
    segs = cv2.HoughLinesP(
        edges, 1, np.pi / 180, hough_threshold,
        minLineLength=min_len, maxLineGap=max_line_gap,
    )
    if segs is None or len(segs) < 2:
        return None
    segs = segs.reshape(-1, 4).astype(float)  # (n, 4): x1,y1,x2,y2

    # Orientation of each segment, doubled so 179 deg and 1 deg are neighbours
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    theta = np.arctan2(dy, dx)
    feats = np.stack([np.cos(2 * theta), np.sin(2 * theta)], axis=1).astype(np.float32)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    _, labels, _ = cv2.kmeans(feats, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    if len(set(labels)) < 2:
        return None

    groups = []
    for k in (0, 1):
        pts = np.concatenate(
            [segs[labels == k][:, :2], segs[labels == k][:, 2:]], axis=0
        ).astype(np.float32)
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        groups.append((line, pts))

    # Orientation difference between groups (0-90 deg); reject near-parallel
    d1, d2 = groups[0][0][:2], groups[1][0][:2]
    sep = math.degrees(math.acos(min(1.0, abs(float(np.dot(d1, d2))))))
    if sep < min_separation_deg:
        return None

    vertex = _line_intersection(groups[0][0], groups[1][0])
    if vertex is None:
        return None
    vx, vy = vertex

    ends, dirs = [], []
    for line, pts in groups:
        d = line[:2].astype(float)
        # orient the fitted direction so it points from the vertex toward the arm
        mean_vec = pts.mean(axis=0) - np.array([vx, vy])
        if np.dot(d, mean_vec) < 0:
            d = -d
        dirs.append(d)
        # tip = projection of the farthest point onto the fitted arm line
        proj = (pts - np.array([vx, vy])) @ d
        length = float(proj.max())
        ends.append((float(vx + d[0] * length), float(vy + d[1] * length)))

    # angle between the two fitted arm directions (more accurate than using
    # edge corner points, which are offset by half the specimen width)
    cosang = float(np.clip(np.dot(dirs[0], dirs[1]), -1.0, 1.0))
    angle = math.degrees(math.acos(cosang))
    return AngleResult(angle, vertex, ends[0], ends[1], "auto")
