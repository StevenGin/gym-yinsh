"""
YINSH game state and logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple
import numpy as np

from gym_yinsh.board import (
    N_CELLS, CellState, COORDS, COORD_TO_IDX, RAY,
    RowInfo, find_rows_of_five, get_valid_ring_moves,
)
from gym_yinsh.actions import (
    encode_place_ring, encode_move_ring, encode_remove_row, encode_remove_ring,
    ACTION_SPACE_SIZE,
)


class Phase(IntEnum):
    PLACE_RING  = 0
    MOVE_RING   = 1
    REMOVE_ROW  = 2
    REMOVE_RING = 3
    DONE        = 4


PLAYER_RING   = [CellState.WHITE_RING,   CellState.BLACK_RING]
PLAYER_MARKER = [CellState.WHITE_MARKER, CellState.BLACK_MARKER]
RINGS_TO_WIN     = 3
RINGS_PER_PLAYER = 5


@dataclass
class GameState:
    board: np.ndarray = field(default_factory=lambda: np.zeros(N_CELLS, dtype=np.int8))
    phase: Phase = Phase.PLACE_RING
    current_player: int = 0      # 0=white, 1=black
    rings_placed: List[int] = field(default_factory=lambda: [0, 0])
    rings_removed: List[int] = field(default_factory=lambda: [0, 0])  # score
    # Pending rows to process (RowInfo objects)
    pending_rows: List[RowInfo] = field(default_factory=list)
    # Player resolving rows (may differ when opponent scores on active player's turn)
    resolving_player: int = 0
    done: bool = False
    winner: Optional[int] = None

    def copy(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            phase=self.phase,
            current_player=self.current_player,
            rings_placed=self.rings_placed.copy(),
            rings_removed=self.rings_removed.copy(),
            pending_rows=[RowInfo(r.start_cell, r.dir_idx, r.cells[:]) for r in self.pending_rows],
            resolving_player=self.resolving_player,
            done=self.done,
            winner=self.winner,
        )


class YinshGame:
    """Encapsulates YINSH rules and state transitions."""

    def __init__(self, state: Optional[GameState] = None):
        self.state = state or GameState()

    # ------------------------------------------------------------------ #
    # Valid actions
    # ------------------------------------------------------------------ #

    def valid_action_list(self) -> List[int]:
        """Returns list of valid flat integer actions for the current state."""
        s = self.state
        if s.done:
            return []

        if s.phase == Phase.PLACE_RING:
            return [encode_place_ring(i) for i in range(N_CELLS) if s.board[i] == CellState.EMPTY]

        if s.phase == Phase.MOVE_RING:
            ring = PLAYER_RING[s.current_player]
            actions = []
            for i in range(N_CELLS):
                if s.board[i] == ring:
                    for dest in get_valid_ring_moves(i, s.board):
                        actions.append(encode_move_ring(i, dest))
            return actions

        if s.phase == Phase.REMOVE_ROW:
            return [encode_remove_row(r.start_cell, r.dir_idx) for r in s.pending_rows]

        if s.phase == Phase.REMOVE_RING:
            ring = PLAYER_RING[s.resolving_player]
            return [encode_remove_ring(i) for i in range(N_CELLS) if s.board[i] == ring]

        return []

    def valid_action_mask(self) -> np.ndarray:
        """Returns bool array of shape (ACTION_SPACE_SIZE,) for valid actions."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for a in self.valid_action_list():
            mask[a] = True
        return mask

    # ------------------------------------------------------------------ #
    # Apply action
    # ------------------------------------------------------------------ #

    def step(self, action: int) -> Tuple[float, bool]:
        """Apply flat integer action. Returns (reward, done)."""
        s = self.state
        if s.done:
            raise ValueError("Game is already over.")

        if s.phase == Phase.PLACE_RING:
            cell = action  # PLACE_RING_OFFSET == 0
            return self._place_ring(cell)

        if s.phase == Phase.MOVE_RING:
            from gym_yinsh.actions import MOVE_RING_OFFSET
            a = action - MOVE_RING_OFFSET
            from_cell, to_cell = a // N_CELLS, a % N_CELLS
            return self._move_ring(from_cell, to_cell)

        if s.phase == Phase.REMOVE_ROW:
            from gym_yinsh.actions import REMOVE_ROW_OFFSET
            a = action - REMOVE_ROW_OFFSET
            start_cell, dir_idx = a // 3, a % 3
            return self._remove_row(start_cell, dir_idx)

        if s.phase == Phase.REMOVE_RING:
            from gym_yinsh.actions import REMOVE_RING_OFFSET
            cell = action - REMOVE_RING_OFFSET
            return self._remove_ring(cell)

        raise ValueError(f"Unknown phase {s.phase}")

    # ------------------------------------------------------------------ #
    # Phase handlers
    # ------------------------------------------------------------------ #

    def _place_ring(self, pos: int) -> Tuple[float, bool]:
        s = self.state
        ring = PLAYER_RING[s.current_player]
        s.board[pos] = ring
        s.rings_placed[s.current_player] += 1

        if sum(s.rings_placed) == 2 * RINGS_PER_PLAYER:
            s.phase = Phase.MOVE_RING
            s.current_player = 0
        else:
            s.current_player = 1 - s.current_player
        return 0.0, False

    def _move_ring(self, from_pos: int, to_pos: int) -> Tuple[float, bool]:
        s = self.state
        ring   = PLAYER_RING[s.current_player]
        marker = PLAYER_MARKER[s.current_player]

        s.board[from_pos] = marker
        self._flip_markers_on_path(from_pos, to_pos)
        s.board[to_pos] = ring

        reward = self._check_and_queue_rows()
        if s.done:
            return reward, True
        if not s.pending_rows:
            s.current_player = 1 - s.current_player
        return reward, False

    def _flip_markers_on_path(self, from_pos: int, to_pos: int):
        s = self.state
        fc, tc = COORDS[from_pos], COORDS[to_pos]
        dq, dr, ds = tc[0]-fc[0], tc[1]-fc[1], tc[2]-fc[2]
        length = max(abs(dq), abs(dr), abs(ds))
        if length == 0:
            return
        step = (dq // length, dr // length, ds // length)
        cur = (fc[0]+step[0], fc[1]+step[1], fc[2]+step[2])
        while cur != tc:
            idx = COORD_TO_IDX[cur]
            s.board[idx] = CellState.flip(s.board[idx])
            cur = (cur[0]+step[0], cur[1]+step[1], cur[2]+step[2])

    def _check_and_queue_rows(self) -> float:
        s = self.state
        white_rows = find_rows_of_five(s.board, CellState.WHITE_MARKER)
        black_rows = find_rows_of_five(s.board, CellState.BLACK_MARKER)

        if not white_rows and not black_rows:
            return 0.0

        my_rows  = white_rows if s.current_player == 0 else black_rows
        opp_rows = black_rows if s.current_player == 0 else white_rows

        if my_rows:
            rows, resolver = my_rows, s.current_player
        else:
            rows, resolver = opp_rows, 1 - s.current_player

        s.resolving_player = resolver
        s.pending_rows = rows

        if len(rows) == 1:
            self._execute_row_removal(rows[0])
            s.pending_rows = []
            if s.done:
                return self._reward_for(s.current_player)
            return self._check_and_queue_rows()
        else:
            s.phase = Phase.REMOVE_ROW
            return 0.0

    def _remove_row(self, start_cell: int, dir_idx: int) -> Tuple[float, bool]:
        s = self.state
        chosen = next(
            (r for r in s.pending_rows if r.start_cell == start_cell and r.dir_idx == dir_idx),
            None,
        )
        assert chosen is not None, f"No pending row ({start_cell}, {dir_idx})"
        self._execute_row_removal(chosen)
        s.pending_rows = []

        if s.done:
            return self._reward_for(s.current_player), True

        reward = self._check_and_queue_rows()
        if s.done:
            return reward, True
        if not s.pending_rows:
            s.phase = Phase.MOVE_RING
            s.current_player = 1 - s.current_player
        return reward, False

    def _execute_row_removal(self, row: RowInfo):
        s = self.state
        for cell in row.cells:
            s.board[cell] = CellState.EMPTY
        s.phase = Phase.REMOVE_RING

        # Auto-remove ring if the player has only one ring left
        ring = PLAYER_RING[s.resolving_player]
        ring_positions = [i for i in range(N_CELLS) if s.board[i] == ring]
        if len(ring_positions) == 1:
            self._execute_ring_removal(ring_positions[0])

    def _remove_ring(self, ring_pos: int) -> Tuple[float, bool]:
        s = self.state
        self._execute_ring_removal(ring_pos)
        if s.done:
            return self._reward_for(s.current_player), True
        reward = self._check_and_queue_rows()
        if s.done:
            return reward, True
        if not s.pending_rows:
            s.phase = Phase.MOVE_RING
            s.current_player = 1 - s.current_player
        return reward, False

    def _execute_ring_removal(self, ring_pos: int):
        s = self.state
        s.board[ring_pos] = CellState.EMPTY
        s.rings_removed[s.resolving_player] += 1
        s.phase = Phase.MOVE_RING

        if s.rings_removed[s.resolving_player] >= RINGS_TO_WIN:
            s.done = True
            s.winner = s.resolving_player
            s.phase = Phase.DONE

    def _reward_for(self, player: int) -> float:
        if not self.state.done:
            return 0.0
        return 1.0 if self.state.winner == player else -1.0

    @property
    def current_player(self) -> int:
        return self.state.current_player

    @property
    def done(self) -> bool:
        return self.state.done
