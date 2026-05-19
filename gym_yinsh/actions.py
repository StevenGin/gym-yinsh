"""
Action space encoding for gym-yinsh.

Action space: Discrete(ACTION_SPACE_SIZE)

Phase            | Range            | Size | Decode
-----------------|------------------|------|----------------------------------------
PLACE_RING       | [0, 84]          |   85 | cell = action
MOVE_RING        | [85, 7309]       | 7225 | from = (action-85)//85, to = (action-85)%85
REMOVE_ROW       | [7310, 7564]     |  255 | start = (action-7310)//3, dir = (action-7310)%3
REMOVE_RING      | [7565, 7649]     |   85 | cell = action - 7565
"""
from gym_yinsh.board import N_CELLS

PLACE_RING_OFFSET  = 0
MOVE_RING_OFFSET   = N_CELLS                           # 85
REMOVE_ROW_OFFSET  = MOVE_RING_OFFSET + N_CELLS ** 2  # 7310  (85 + 85*85)
REMOVE_RING_OFFSET = REMOVE_ROW_OFFSET + N_CELLS * 3  # 7565  (+ 85*3)
ACTION_SPACE_SIZE  = REMOVE_RING_OFFSET + N_CELLS      # 7650  (+ 85)


def encode_place_ring(cell: int) -> int:
    return PLACE_RING_OFFSET + cell


def encode_move_ring(from_cell: int, to_cell: int) -> int:
    return MOVE_RING_OFFSET + from_cell * N_CELLS + to_cell


def encode_remove_row(start_cell: int, dir_idx: int) -> int:
    """dir_idx in 0..2 (canonical row directions only)."""
    return REMOVE_ROW_OFFSET + start_cell * 3 + dir_idx


def encode_remove_ring(cell: int) -> int:
    return REMOVE_RING_OFFSET + cell


def decode_action(action: int):
    """Returns (phase_name, *args) tuple."""
    if action < MOVE_RING_OFFSET:
        return "place_ring", action
    if action < REMOVE_ROW_OFFSET:
        a = action - MOVE_RING_OFFSET
        return "move_ring", a // N_CELLS, a % N_CELLS
    if action < REMOVE_RING_OFFSET:
        a = action - REMOVE_ROW_OFFSET
        return "remove_row", a // 3, a % 3
    return "remove_ring", action - REMOVE_RING_OFFSET
