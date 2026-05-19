"""
Pygame renderer for YINSH.

Usage:
    from gym_yinsh.rendering.pygame_renderer import PygameRenderer
    renderer = PygameRenderer()
    renderer.render(game_state)
    renderer.close()

Or just set render_mode="human" on the env:
    env = gym.make("yinsh-v0", render_mode="human")
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from gym_yinsh.board import N_CELLS, CellState, NEIGHBORS
from gym_yinsh.game import GameState, Phase
from gym_yinsh.rendering.geometry import all_pixels

_SCALE = 48.0
_W, _H = 900, 940   # extra height for HUD
_CX, _CY = _W / 2, _H / 2 - 20

_BOARD_BG         = (56, 38, 20)
_LINE_COLOR       = (115, 89, 51)
_EMPTY_DOT        = (90, 72, 46)

_WHITE_RING_EDGE  = (242, 242, 230)
_BLACK_RING_EDGE  = (26, 26, 26)
_WHITE_MARKER     = (235, 235, 220)
_BLACK_MARKER     = (30, 30, 30)

_RING_RADIUS    = int(_SCALE * 0.38)
_RING_LW        = max(2, int(_SCALE * 0.12))
_MARKER_RADIUS  = int(_SCALE * 0.22)
_DOT_RADIUS     = max(2, int(_SCALE * 0.07))

_FPS = 30


class PygameRenderer:
    def __init__(self, caption: str = "YINSH", fps: int = _FPS):
        self._pixels: Optional[np.ndarray] = None
        self._screen = None
        self._clock = None
        self._font = None
        self._caption = caption
        self._fps = fps
        self._initialised = False

    def _init(self):
        if self._initialised:
            return
        import pygame
        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((_W, _H))
        pygame.display.set_caption(self._caption)
        self._clock = pygame.time.Clock()
        try:
            self._font = pygame.font.SysFont("monospace", 14)
        except Exception:
            self._font = pygame.font.Font(None, 16)
        self._pixels = all_pixels(_SCALE, _CX, _CY)
        self._initialised = True

    def render(self, state: GameState) -> bool:
        """
        Draw current state. Returns False if the window was closed.
        Call in your game loop after each step.
        """
        self._init()
        pg = self._pygame

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return False

        screen = self._screen
        screen.fill(_BOARD_BG)

        self._draw_grid(screen)
        self._draw_pieces(screen, state)
        self._draw_hud(screen, state)

        pg.display.flip()
        self._clock.tick(self._fps)
        return True

    def wait_for_close(self):
        """Block until the user closes the window."""
        if not self._initialised:
            return
        pg = self._pygame
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    return
            self._clock.tick(30)

    def close(self):
        if self._initialised:
            self._pygame.quit()
            self._initialised = False

    # ------------------------------------------------------------------ #

    def _draw_grid(self, screen):
        pg = self._pygame
        px = self._pixels
        drawn: set = set()
        for i in range(N_CELLS):
            for nb, _ in NEIGHBORS[i]:
                key = (min(i, nb), max(i, nb))
                if key in drawn:
                    continue
                drawn.add(key)
                x0, y0 = int(px[i, 0]), int(px[i, 1])
                x1, y1 = int(px[nb, 0]), int(px[nb, 1])
                pg.draw.line(screen, _LINE_COLOR, (x0, y0), (x1, y1), 1)

        for i in range(N_CELLS):
            x, y = int(px[i, 0]), int(px[i, 1])
            pg.draw.circle(screen, _EMPTY_DOT, (x, y), _DOT_RADIUS)

    def _draw_pieces(self, screen, state: GameState):
        pg = self._pygame
        px = self._pixels
        for i in range(N_CELLS):
            v = state.board[i]
            x, y = int(px[i, 0]), int(px[i, 1])

            if v == CellState.WHITE_RING:
                pg.draw.circle(screen, _WHITE_RING_EDGE, (x, y), _RING_RADIUS, _RING_LW)
            elif v == CellState.BLACK_RING:
                pg.draw.circle(screen, _BLACK_RING_EDGE, (x, y), _RING_RADIUS, _RING_LW)
                # Inner lighter outline for visibility on dark bg
                pg.draw.circle(screen, _LINE_COLOR, (x, y), _RING_RADIUS, 1)
            elif v == CellState.WHITE_MARKER:
                pg.draw.circle(screen, _WHITE_MARKER, (x, y), _MARKER_RADIUS)
            elif v == CellState.BLACK_MARKER:
                pg.draw.circle(screen, _BLACK_MARKER, (x, y), _MARKER_RADIUS)
                pg.draw.circle(screen, _LINE_COLOR, (x, y), _MARKER_RADIUS, 1)

    def _draw_hud(self, screen, state: GameState):
        pg = self._pygame
        score_w, score_b = state.rings_removed
        rings_w = int((screen.get_width() - 0) // 1)  # unused

        if state.done:
            winner = "White" if state.winner == 0 else "Black"
            line1 = f"Game Over  —  {winner} wins!"
            line2 = f"White rings removed: {score_w}   Black rings removed: {score_b}"
        else:
            phase_name = state.phase.name
            player = "White" if state.current_player == 0 else "Black"
            line1 = f"Phase: {phase_name}    Turn: {player}"
            line2 = f"White: {score_w}/3 rings scored    Black: {score_b}/3 rings scored"

        col = (230, 215, 180)
        surf1 = self._font.render(line1, True, col)
        surf2 = self._font.render(line2, True, col)
        screen.blit(surf1, (_W // 2 - surf1.get_width() // 2, _H - 38))
        screen.blit(surf2, (_W // 2 - surf2.get_width() // 2, _H - 20))
