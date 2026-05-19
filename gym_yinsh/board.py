"""
YINSH board geometry using cube coordinates (q, r, s) where q+r+s=0.

The board is a radius-5 hexagon (91 cells) minus 6 corner cells = 85 cells.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Set

# Cube coordinate type
Coord = Tuple[int, int, int]

# Six movement directions in cube coordinates
DIRECTIONS: List[Coord] = [
    (1, -1, 0),   # E
    (1, 0, -1),   # NE
    (0, 1, -1),   # NW
    (-1, 1, 0),   # W
    (-1, 0, 1),   # SW
    (0, -1, 1),   # SE
]

# The 6 corner positions of radius-5 hex that YINSH removes
_REMOVED_CORNERS: Set[Coord] = {
    (5, -5, 0), (-5, 5, 0),
    (5, 0, -5), (-5, 0, 5),
    (0, 5, -5), (0, -5, 5),
}


def _build_board() -> Tuple[List[Coord], dict]:
    """Build sorted list of valid positions and coord→index mapping."""
    coords = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            s = -q - r
            if abs(s) > 5:
                continue
            coord = (q, r, s)
            if coord in _REMOVED_CORNERS:
                continue
            coords.append(coord)
    coords.sort()
    coord_to_idx = {c: i for i, c in enumerate(coords)}
    return coords, coord_to_idx


COORDS: List[Coord] = []
COORD_TO_IDX: dict = {}
COORDS, COORD_TO_IDX = _build_board()
N_CELLS: int = len(COORDS)  # 85

assert N_CELLS == 85, f"Expected 85 cells, got {N_CELLS}"

# Precompute neighbors for each cell: neighbors[i] = list of (neighbor_idx, dir_idx)
NEIGHBORS: List[List[Tuple[int, int]]] = [[] for _ in range(N_CELLS)]
for _i, _coord in enumerate(COORDS):
    for _d, _dir in enumerate(DIRECTIONS):
        _nb = (_coord[0] + _dir[0], _coord[1] + _dir[1], _coord[2] + _dir[2])
        if _nb in COORD_TO_IDX:
            NEIGHBORS[_i].append((COORD_TO_IDX[_nb], _d))

# Precompute all cells along each ray from each cell:
# RAY[cell_idx][dir_idx] = [idx1, idx2, ...] in order from cell outward (not including cell itself)
RAY: List[List[List[int]]] = [[[] for _ in DIRECTIONS] for _ in range(N_CELLS)]
for _i, _coord in enumerate(COORDS):
    for _d, _dir in enumerate(DIRECTIONS):
        ray = []
        cur = (_coord[0] + _dir[0], _coord[1] + _dir[1], _coord[2] + _dir[2])
        while cur in COORD_TO_IDX:
            ray.append(COORD_TO_IDX[cur])
            cur = (cur[0] + _dir[0], cur[1] + _dir[1], cur[2] + _dir[2])
        RAY[_i][_d] = ray


def get_valid_ring_moves(cell: int, board: np.ndarray) -> List[int]:
    """
    Returns list of valid destination cell indices for a ring at `cell`.

    Movement rules:
    - In each of 6 directions, scan along the ray.
    - If no markers are encountered before hitting a ring/edge: can land on any
      empty cell in that direction (up to but not including the first ring/edge).
    - If markers are encountered: must land on the first empty cell after the
      last consecutive block of markers.
    """
    destinations = []
    for d in range(6):
        ray = RAY[cell][d]
        found_marker = False
        last_marker_end = -1  # index in `ray` after the last marker block

        i = 0
        while i < len(ray):
            c = ray[i]
            cell_val = board[c]
            if cell_val == CellState.EMPTY:
                if found_marker:
                    # Must stop here (first empty after marker block)
                    destinations.append(c)
                    break
                else:
                    # Can stop anywhere before first marker
                    destinations.append(c)
                    i += 1
            elif cell_val in (CellState.WHITE_MARKER, CellState.BLACK_MARKER):
                found_marker = True
                i += 1
            else:
                # Hit a ring: blocked
                break

    return destinations


class CellState:
    EMPTY = 0
    WHITE_RING = 1
    BLACK_RING = 2
    WHITE_MARKER = 3
    BLACK_MARKER = 4

    @staticmethod
    def is_ring(v: int) -> bool:
        return v in (CellState.WHITE_RING, CellState.BLACK_RING)

    @staticmethod
    def is_marker(v: int) -> bool:
        return v in (CellState.WHITE_MARKER, CellState.BLACK_MARKER)

    @staticmethod
    def flip(v: int) -> int:
        """Flip a marker to the opposite color."""
        if v == CellState.WHITE_MARKER:
            return CellState.BLACK_MARKER
        if v == CellState.BLACK_MARKER:
            return CellState.WHITE_MARKER
        return v


class RowInfo:
    """A row of 5 consecutive same-color markers, identified by canonical (start, dir)."""
    __slots__ = ("start_cell", "dir_idx", "cells")

    def __init__(self, start_cell: int, dir_idx: int, cells: List[int]):
        self.start_cell = start_cell  # first cell in canonical direction
        self.dir_idx = dir_idx        # 0, 1, or 2
        self.cells = cells            # ordered list of 5 cell indices


def find_rows_of_five(board: np.ndarray, color_marker: int) -> List[RowInfo]:
    """
    Find all rows of 5 consecutive markers of `color_marker`.
    A run of 6 yields two overlapping RowInfo entries.
    Each row is uniquely identified by (start_cell, dir_idx).
    """
    rows: List[RowInfo] = []
    visited_lines: Set[Tuple[int, int]] = set()

    for origin in range(N_CELLS):
        for d in range(3):  # 3 canonical directions cover all lines without duplication
            if (origin, d) in visited_lines:
                continue

            # Build full line through origin in both directions along axis d
            bwd = RAY[origin][d + 3]          # cells in opposite direction
            full_line = list(reversed(bwd)) + [origin] + RAY[origin][d]

            # Mark all cells on this line as visited for direction d
            for c in full_line:
                visited_lines.add((c, d))

            # Scan for runs of color_marker
            run_start: Optional[int] = None
            for j, c in enumerate(full_line):
                if board[c] == color_marker:
                    if run_start is None:
                        run_start = j
                else:
                    if run_start is not None:
                        _emit_rows(rows, full_line, run_start, j, d)
                        run_start = None
            if run_start is not None:
                _emit_rows(rows, full_line, run_start, len(full_line), d)

    return rows


def _emit_rows(
    rows: List[RowInfo],
    line: List[int],
    run_start: int,
    run_end: int,
    dir_idx: int,
):
    run_len = run_end - run_start
    for k in range(run_len - 4):
        cells = line[run_start + k: run_start + k + 5]
        rows.append(RowInfo(start_cell=cells[0], dir_idx=dir_idx, cells=cells))
