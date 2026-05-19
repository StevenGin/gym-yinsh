"""Tests for board geometry."""
import numpy as np
import pytest
from gym_yinsh.board import (
    N_CELLS, COORDS, COORD_TO_IDX, RAY, DIRECTIONS,
    CellState, find_rows_of_five, get_valid_ring_moves,
)


def test_board_size():
    assert N_CELLS == 85


def test_no_corners():
    corners = {(5,-5,0), (-5,5,0), (5,0,-5), (-5,0,5), (0,5,-5), (0,-5,5)}
    for c in corners:
        assert c not in COORD_TO_IDX


def test_all_coords_valid():
    for q, r, s in COORDS:
        assert q + r + s == 0
        assert max(abs(q), abs(r), abs(s)) <= 5


def test_ray_cells_in_board():
    for i in range(N_CELLS):
        for d in range(6):
            for cell in RAY[i][d]:
                assert 0 <= cell < N_CELLS


def test_ray_direction_consistent():
    """Each step along a ray moves by exactly one direction step."""
    for i in range(N_CELLS):
        for d, direction in enumerate(DIRECTIONS):
            ray = RAY[i][d]
            prev = COORDS[i]
            for cell in ray:
                coord = COORDS[cell]
                assert coord == (prev[0]+direction[0], prev[1]+direction[1], prev[2]+direction[2])
                prev = coord


def test_find_rows_none():
    board = np.zeros(N_CELLS, dtype=np.int8)
    assert find_rows_of_five(board, CellState.WHITE_MARKER) == []


def test_find_rows_exactly_five():
    board = np.zeros(N_CELLS, dtype=np.int8)
    start = 0
    cells = [start] + RAY[start][0][:4]
    assert len(cells) == 5
    for c in cells:
        board[c] = CellState.WHITE_MARKER
    rows = find_rows_of_five(board, CellState.WHITE_MARKER)
    assert len(rows) == 1
    assert sorted(rows[0].cells) == sorted(cells)


def test_find_rows_six_gives_two():
    board = np.zeros(N_CELLS, dtype=np.int8)
    start = 0
    ray = RAY[start][0]
    if len(ray) < 5:
        pytest.skip("Ray too short")
    cells = [start] + ray[:5]
    for c in cells:
        board[c] = CellState.WHITE_MARKER
    rows = find_rows_of_five(board, CellState.WHITE_MARKER)
    assert len(rows) == 2


def test_flip_marker():
    assert CellState.flip(CellState.WHITE_MARKER) == CellState.BLACK_MARKER
    assert CellState.flip(CellState.BLACK_MARKER) == CellState.WHITE_MARKER
    assert CellState.flip(CellState.EMPTY) == CellState.EMPTY


def test_ring_moves_empty_board():
    """Ring on empty board can move in all rays."""
    board = np.zeros(N_CELLS, dtype=np.int8)
    center = COORD_TO_IDX[(0, 0, 0)]
    board[center] = CellState.WHITE_RING
    moves = get_valid_ring_moves(center, board)
    # Should be able to reach all cells in each of 6 directions
    assert len(moves) > 0
    # All destinations should be empty
    for m in moves:
        assert board[m] == CellState.EMPTY


def test_ring_blocked_by_ring():
    """Ring cannot jump over another ring."""
    board = np.zeros(N_CELLS, dtype=np.int8)
    center = COORD_TO_IDX[(0, 0, 0)]
    board[center] = CellState.WHITE_RING
    ray0 = RAY[center][0]
    if len(ray0) >= 2:
        blocking = ray0[0]
        board[blocking] = CellState.BLACK_RING
        moves = get_valid_ring_moves(center, board)
        # Blocking ring's direction should produce no moves in that direction
        assert blocking not in moves


def test_ring_jumps_markers():
    """Ring must land on first empty after consecutive markers."""
    board = np.zeros(N_CELLS, dtype=np.int8)
    center = COORD_TO_IDX[(0, 0, 0)]
    board[center] = CellState.WHITE_RING
    ray0 = RAY[center][0]
    if len(ray0) < 3:
        pytest.skip("Ray too short")
    # Place markers at ray0[0] and ray0[1]; ring must land on ray0[2]
    board[ray0[0]] = CellState.WHITE_MARKER
    board[ray0[1]] = CellState.WHITE_MARKER
    moves = get_valid_ring_moves(center, board)
    assert ray0[2] in moves
    # Cannot land on the markers themselves
    assert ray0[0] not in moves
    assert ray0[1] not in moves
