"""
utils.py
--------
Drawing helpers and result bookkeeping for the crease recovery app.
"""

from typing import List

import cv2
import numpy as np
import pandas as pd

from angle_detection import AngleResult


def annotate(img: np.ndarray, result: AngleResult) -> np.ndarray:
    """Draw the two arms, the vertex and the angle value on a copy of the image."""
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, int(round(max(h, w) / 400)))
    vx, vy = map(int, result.vertex)
    e1 = tuple(map(int, result.end1))
    e2 = tuple(map(int, result.end2))

    cv2.line(out, (vx, vy), e1, (0, 200, 0), thick)
    cv2.line(out, (vx, vy), e2, (0, 120, 255), thick)
    cv2.circle(out, (vx, vy), thick * 3, (0, 0, 255), -1)
    cv2.circle(out, e1, thick * 2, (0, 200, 0), -1)
    cv2.circle(out, e2, thick * 2, (0, 120, 255), -1)

    # Arc showing the measured angle
    a1 = np.degrees(np.arctan2(e1[1] - vy, e1[0] - vx))
    a2 = np.degrees(np.arctan2(e2[1] - vy, e2[0] - vx))
    diff = (a2 - a1 + 180) % 360 - 180
    start, end = (a1, a1 + diff) if diff >= 0 else (a1 + diff, a1)
    radius = int(0.12 * max(h, w))
    cv2.ellipse(out, (vx, vy), (radius, radius), 0, start, end, (255, 0, 0), thick)

    label = f"{result.angle:.1f} deg"
    scale = max(0.7, max(h, w) / 900)
    cv2.putText(out, label, (vx + radius + 10, max(30, vy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 0, 0), thick, cv2.LINE_AA)
    return out


def draw_points(img: np.ndarray, points: List[tuple]) -> np.ndarray:
    """Draw the points clicked so far (vertex = red, arm tips = green / orange)."""
    out = img.copy()
    colors = [(0, 0, 255), (0, 200, 0), (0, 120, 255)]
    r = max(4, int(max(out.shape[:2]) / 150))
    for i, (x, y) in enumerate(points[:3]):
        cv2.circle(out, (int(x), int(y)), r, colors[i], -1)
        cv2.putText(out, str(i + 1), (int(x) + r + 2, int(y) - r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2, cv2.LINE_AA)
    if len(points) == 3:
        cv2.line(out, tuple(map(int, points[0])), tuple(map(int, points[1])), (0, 200, 0), 2)
        cv2.line(out, tuple(map(int, points[0])), tuple(map(int, points[2])), (0, 120, 255), 2)
    return out


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std per direction plus total (warp + weft) crease recovery angle."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("Direction")["Angle (deg)"].agg(["count", "mean", "std"]).round(2)
    g.columns = ["n", "Mean (deg)", "Std dev (deg)"]
    if {"Warp", "Weft"}.issubset(g.index):
        total = g.loc["Warp", "Mean (deg)"] + g.loc["Weft", "Mean (deg)"]
        g.loc["Total (Warp + Weft)"] = [None, round(total, 2), None]
    return g
