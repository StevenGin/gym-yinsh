"""
Matplotlib renderer for YINSH.

Usage:
    from gym_yinsh.rendering.matplotlib_renderer import MatplotlibRenderer
    renderer = MatplotlibRenderer()
    renderer.render(game_state)   # pops up / updates a figure
    renderer.save("frame.png")    # save last frame
    renderer.close()
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch

from gym_yinsh.board import N_CELLS, CellState, NEIGHBORS
from gym_yinsh.game import GameState, Phase
from gym_yinsh.rendering.geometry import all_pixels

_SCALE = 48.0
_W, _H = 900, 900
_CX, _CY = _W / 2, _H / 2

# Colours (R, G, B) normalised 0-1
_BOARD_BG   = (0.22, 0.15, 0.08)
_LINE_COLOR = (0.45, 0.35, 0.20)
_EMPTY_DOT  = (0.35, 0.28, 0.18)

_WHITE_RING_EDGE  = (0.95, 0.95, 0.90)
_WHITE_RING_FACE  = _BOARD_BG          # hollow
_BLACK_RING_EDGE  = (0.10, 0.10, 0.10)
_BLACK_RING_FACE  = _BOARD_BG

_WHITE_MARKER_FACE = (0.92, 0.92, 0.86)
_BLACK_MARKER_FACE = (0.12, 0.12, 0.12)

_RING_RADIUS    = _SCALE * 0.38
_RING_LW        = _SCALE * 0.12
_MARKER_RADIUS  = _SCALE * 0.22
_DOT_RADIUS     = _SCALE * 0.06


class MatplotlibRenderer:
    def __init__(self, figsize: float = 7.0):
        self._pixels = all_pixels(_SCALE, _CX, _CY)
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._figsize = figsize

    def render(self, state: GameState) -> np.ndarray:
        """
        Draw the board and return an RGB array (H, W, 3) uint8.
        Also updates the live figure if plt.isinteractive().
        """
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(self._figsize, self._figsize))
            self._fig.patch.set_facecolor(_BOARD_BG)

        ax = self._ax
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(0, _W)
        ax.set_ylim(0, _H)
        ax.axis("off")
        ax.set_facecolor(_BOARD_BG)
        self._fig.patch.set_facecolor(_BOARD_BG)

        self._draw_grid(ax)
        self._draw_pieces(ax, state)
        self._draw_hud(ax, state)

        self._fig.tight_layout(pad=0)
        self._fig.canvas.draw()

        # Convert to RGB array
        buf = self._fig.canvas.buffer_rgba()
        img = np.asarray(buf)[..., :3]
        return img

    def show(self, state: GameState):
        """Render and display interactively (blocks until window closed)."""
        self.render(state)
        plt.show()

    def save(self, path: str, state: Optional[GameState] = None):
        if state is not None:
            self.render(state)
        self._fig.savefig(path, bbox_inches="tight", facecolor=_BOARD_BG)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    # ------------------------------------------------------------------ #

    def _draw_grid(self, ax: plt.Axes):
        px = self._pixels
        drawn: set = set()
        for i in range(N_CELLS):
            for nb, _ in NEIGHBORS[i]:
                key = (min(i, nb), max(i, nb))
                if key in drawn:
                    continue
                drawn.add(key)
                x0, y0 = px[i]
                x1, y1 = px[nb]
                ax.plot([x0, x1], [_H - y0, _H - y1],
                        color=_LINE_COLOR, lw=0.8, zorder=1)

        # Empty cell dots
        for i in range(N_CELLS):
            x, y = px[i]
            c = Circle((x, _H - y), _DOT_RADIUS, color=_EMPTY_DOT, zorder=2)
            ax.add_patch(c)

    def _draw_pieces(self, ax: plt.Axes, state: GameState):
        px = self._pixels
        for i in range(N_CELLS):
            v = state.board[i]
            x, y = px[i]
            sy = _H - y  # flip Y for matplotlib

            if v == CellState.WHITE_RING:
                c = Circle((x, sy), _RING_RADIUS,
                            facecolor=_WHITE_RING_FACE,
                            edgecolor=_WHITE_RING_EDGE,
                            linewidth=_RING_LW, zorder=4)
                ax.add_patch(c)

            elif v == CellState.BLACK_RING:
                c = Circle((x, sy), _RING_RADIUS,
                            facecolor=_BLACK_RING_FACE,
                            edgecolor=_BLACK_RING_EDGE,
                            linewidth=_RING_LW, zorder=4)
                ax.add_patch(c)

            elif v == CellState.WHITE_MARKER:
                c = Circle((x, sy), _MARKER_RADIUS,
                            facecolor=_WHITE_MARKER_FACE,
                            edgecolor=_WHITE_RING_EDGE,
                            linewidth=0.5, zorder=3)
                ax.add_patch(c)

            elif v == CellState.BLACK_MARKER:
                c = Circle((x, sy), _MARKER_RADIUS,
                            facecolor=_BLACK_MARKER_FACE,
                            edgecolor=_BLACK_RING_EDGE,
                            linewidth=0.5, zorder=3)
                ax.add_patch(c)

    def _draw_hud(self, ax: plt.Axes, state: GameState):
        phase_name = state.phase.name
        player = "White" if state.current_player == 0 else "Black"
        score_w, score_b = state.rings_removed

        if state.done:
            winner = "White" if state.winner == 0 else "Black"
            hud = f"Game Over  —  {winner} wins!"
        else:
            hud = f"Phase: {phase_name}   Turn: {player}   W: {score_w}/3   B: {score_b}/3"

        ax.text(_W / 2, _H - 14, hud,
                ha="center", va="top",
                color=(0.9, 0.85, 0.75),
                fontsize=11, fontfamily="monospace",
                zorder=10)
