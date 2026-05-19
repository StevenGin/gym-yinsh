"""
Gymnasium environment for YINSH.

Observation space (Dict):
  board:          (4, N_CELLS) int8   — binary channels: [P0_rings, P1_rings, P0_markers, P1_markers]
  context:        (8,) int8           — [phase, current_player, score_p0, score_p1,
                                         rings_on_board_p0, rings_on_board_p1,
                                         rings_placed_p0, rings_placed_p1]

Action space: Discrete(ACTION_SPACE_SIZE=7650)
  See gym_yinsh/actions.py for encoding details.
  Action mask is returned in info["action_mask"] each step.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_yinsh.board import N_CELLS, CellState
from gym_yinsh.game import GameState, YinshGame, Phase, RINGS_TO_WIN, RINGS_PER_PLAYER
from gym_yinsh.actions import ACTION_SPACE_SIZE


class YinshEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 1, shape=(4, N_CELLS), dtype=np.int8),
            "context": spaces.Box(
                low=np.zeros(8, dtype=np.int8),
                high=np.array([4, 1, 3, 3, 5, 5, 5, 5], dtype=np.int8),
                dtype=np.int8,
            ),
        })

        self.game: Optional[YinshGame] = None

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.game = YinshGame()
        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert self.game is not None, "Call reset() before step()"
        action = int(action)

        mask = self.game.valid_action_mask()
        if not mask[action]:
            obs = self._get_obs()
            return obs, -1.0, True, False, self._get_info()

        reward, done = self.game.step(action)
        obs = self._get_obs()
        return obs, float(reward), done, False, self._get_info()

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Observation / info builders
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> Dict:
        s = self.game.state

        board = np.zeros((4, N_CELLS), dtype=np.int8)
        for i in range(N_CELLS):
            v = s.board[i]
            if v == CellState.WHITE_RING:
                board[0, i] = 1
            elif v == CellState.BLACK_RING:
                board[1, i] = 1
            elif v == CellState.WHITE_MARKER:
                board[2, i] = 1
            elif v == CellState.BLACK_MARKER:
                board[3, i] = 1

        rings_on_board = [
            int((s.board == CellState.WHITE_RING).sum()),
            int((s.board == CellState.BLACK_RING).sum()),
        ]

        context = np.array([
            int(s.phase),
            s.current_player,
            s.rings_removed[0],
            s.rings_removed[1],
            rings_on_board[0],
            rings_on_board[1],
            s.rings_placed[0],
            s.rings_placed[1],
        ], dtype=np.int8)

        return {"board": board, "context": context}

    def _get_info(self) -> Dict:
        s = self.game.state
        return {
            "phase": s.phase,
            "current_player": s.current_player,
            "rings_removed": s.rings_removed.copy(),
            "winner": s.winner,
            "action_mask": self.game.valid_action_mask(),
        }

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #

    def _render_ansi(self) -> str:
        from gym_yinsh.board import COORDS
        s = self.game.state

        symbols = {
            CellState.EMPTY:        ".",
            CellState.WHITE_RING:   "O",
            CellState.BLACK_RING:   "X",
            CellState.WHITE_MARKER: "o",
            CellState.BLACK_MARKER: "x",
        }

        header = (
            f"Phase: {s.phase.name}  "
            f"Player: {'W' if s.current_player == 0 else 'B'}  "
            f"Score W:{s.rings_removed[0]} B:{s.rings_removed[1]}"
        )

        coord_to_sym = {COORDS[i]: symbols[s.board[i]] for i in range(N_CELLS)}
        q_vals = sorted(set(c[0] for c in COORDS))
        r_vals = sorted(set(c[1] for c in COORDS))

        lines = [header]
        for r in r_vals:
            cells = [coord_to_sym[(q, r, -q-r)] for q in q_vals if (q, r, -q-r) in coord_to_sym]
            indent = " " * (5 + r)
            lines.append(indent + " ".join(cells))

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Action mask helper (sb3-contrib MaskablePPO compatible)
    # ------------------------------------------------------------------ #

    def action_masks(self) -> np.ndarray:
        if self.game is None:
            return np.ones(ACTION_SPACE_SIZE, dtype=bool)
        return self.game.valid_action_mask()
