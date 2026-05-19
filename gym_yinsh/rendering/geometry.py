"""
Shared geometry for rendering: cube coords → pixel positions.

Flat-top hex layout:
  px = scale * 3/2 * q
  py = scale * (sqrt(3)/2 * q + sqrt(3) * r)
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from gym_yinsh.board import COORDS, N_CELLS, NEIGHBORS

_SQRT3 = math.sqrt(3)


def cell_pixel(cell_idx: int, scale: float, cx: float, cy: float) -> Tuple[float, float]:
    q, r, _ = COORDS[cell_idx]
    x = cx + scale * 1.5 * q
    y = cy + scale * (_SQRT3 / 2 * q + _SQRT3 * r)
    return x, y


def all_pixels(scale: float, cx: float, cy: float) -> np.ndarray:
    """Returns (N_CELLS, 2) float array of pixel positions."""
    pts = np.empty((N_CELLS, 2), dtype=float)
    for i, (q, r, _) in enumerate(COORDS):
        pts[i, 0] = cx + scale * 1.5 * q
        pts[i, 1] = cy + scale * (_SQRT3 / 2 * q + _SQRT3 * r)
    return pts
