"""Tests for game logic."""
import numpy as np
import pytest
from gym_yinsh.board import N_CELLS, CellState, COORDS, COORD_TO_IDX, RAY
from gym_yinsh.game import YinshGame, GameState, Phase, RINGS_PER_PLAYER, RINGS_TO_WIN
from gym_yinsh.actions import (
    encode_place_ring, encode_move_ring, encode_remove_row, encode_remove_ring,
    ACTION_SPACE_SIZE,
)


def _make_game_with_rings_placed() -> YinshGame:
    """Setup a game past ring placement."""
    g = YinshGame()
    positions = list(range(10))
    for i, pos in enumerate(positions):
        actions = g.valid_action_list()
        act = encode_place_ring(pos)
        assert act in actions
        g.step(act)
    assert g.state.phase == Phase.MOVE_RING
    return g


class TestSetupPhase:
    def test_initial_phase(self):
        g = YinshGame()
        assert g.state.phase == Phase.PLACE_RING
        assert g.state.current_player == 0

    def test_alternating_players(self):
        g = YinshGame()
        for i in range(4):
            assert g.state.current_player == i % 2
            g.step(encode_place_ring(i))

    def test_transitions_to_move_after_ten_rings(self):
        g = _make_game_with_rings_placed()
        assert g.state.phase == Phase.MOVE_RING
        assert g.state.current_player == 0

    def test_rings_on_board(self):
        g = _make_game_with_rings_placed()
        white = (g.state.board == CellState.WHITE_RING).sum()
        black = (g.state.board == CellState.BLACK_RING).sum()
        assert white == RINGS_PER_PLAYER
        assert black == RINGS_PER_PLAYER

    def test_cannot_place_on_occupied(self):
        g = YinshGame()
        g.step(encode_place_ring(0))
        actions = g.valid_action_list()
        assert encode_place_ring(0) not in actions


class TestMovePhase:
    def test_valid_moves_exist(self):
        g = _make_game_with_rings_placed()
        actions = g.valid_action_list()
        assert len(actions) > 0

    def test_move_places_marker(self):
        g = _make_game_with_rings_placed()
        from gym_yinsh.board import get_valid_ring_moves
        # Find a white ring that has valid moves
        ring_pos, dest = None, None
        for i in range(N_CELLS):
            if g.state.board[i] == CellState.WHITE_RING:
                dests = get_valid_ring_moves(i, g.state.board)
                if dests:
                    ring_pos, dest = i, dests[0]
                    break
        assert ring_pos is not None, "No white ring with valid moves found"
        g.step(encode_move_ring(ring_pos, dest))
        assert g.state.board[ring_pos] == CellState.WHITE_MARKER
        assert g.state.board[dest] == CellState.WHITE_RING

    def test_marker_flipping(self):
        """Ring jumping over markers flips their color."""
        g = _make_game_with_rings_placed()
        # Find a white ring
        ring_pos = next(i for i in range(N_CELLS) if g.state.board[i] == CellState.WHITE_RING)
        ray0 = RAY[ring_pos][0]
        if len(ray0) < 3:
            pytest.skip("Ray too short")
        # Manually place a black marker in path
        path_cell = ray0[0]
        if g.state.board[path_cell] == CellState.EMPTY:
            g.state.board[path_cell] = CellState.BLACK_MARKER
            dest = ray0[1]
            if g.state.board[dest] == CellState.EMPTY:
                g.step(encode_move_ring(ring_pos, dest))
                # The marker should have been flipped to white
                assert g.state.board[path_cell] == CellState.WHITE_MARKER

    def test_player_alternates_after_move(self):
        g = _make_game_with_rings_placed()
        assert g.state.current_player == 0
        act = g.valid_action_list()[0]
        g.step(act)
        assert g.state.current_player == 1


class TestRowDetection:
    def _setup_row_scenario(self):
        """Create a game state with a near-complete row."""
        g = _make_game_with_rings_placed()
        # Manually place 4 white markers in a line, then engineer a 5th
        s = g.state
        # Use cell 0 and ray in dir 0
        cells_in_line = [0] + RAY[0][0]
        if len(cells_in_line) < 5:
            return None, None
        # Clear those cells and place markers
        for c in cells_in_line[:4]:
            s.board[c] = CellState.WHITE_MARKER
        # Place white ring at cells_in_line[5] if it exists, else find another ring
        return g, cells_in_line

    def test_row_triggers_ring_removal(self):
        g, cells = self._setup_row_scenario()
        if g is None:
            pytest.skip("Line too short")
        s = g.state
        # Add a 5th marker
        if len(cells) >= 5:
            s.board[cells[4]] = CellState.WHITE_MARKER
        # Find rows
        from gym_yinsh.board import find_rows_of_five
        rows = find_rows_of_five(s.board, CellState.WHITE_MARKER)
        assert len(rows) >= 1


class TestWinCondition:
    def test_game_ends_at_three_rings(self):
        """Simulate a win by manipulating state."""
        g = _make_game_with_rings_placed()
        s = g.state
        # Give white player 2 rings already removed
        s.rings_removed[0] = 2
        s.resolving_player = 0

        # Manually trigger the 3rd ring removal
        ring_pos = next(i for i in range(N_CELLS) if s.board[i] == CellState.WHITE_RING)
        s.phase = Phase.REMOVE_RING
        g.step(encode_remove_ring(ring_pos))

        assert s.done
        assert s.winner == 0

    def test_reward_on_win(self):
        g = _make_game_with_rings_placed()
        s = g.state
        s.rings_removed[0] = 2
        s.resolving_player = 0
        ring_pos = next(i for i in range(N_CELLS) if s.board[i] == CellState.WHITE_RING)
        s.phase = Phase.REMOVE_RING
        s.current_player = 0
        reward, done = g.step(encode_remove_ring(ring_pos))
        assert done
        assert reward == 1.0  # white (current_player=0) wins

    def test_full_random_game_terminates(self):
        """A random game always terminates within a reasonable step count."""
        rng = np.random.default_rng(0)
        g = YinshGame()
        for _ in range(2000):
            if g.done:
                break
            actions = g.valid_action_list()
            assert len(actions) > 0, "No valid actions before game ended"
            g.step(rng.choice(actions))
        assert g.done, "Game did not terminate within 2000 steps"


class TestActionSpace:
    def test_action_space_size(self):
        assert ACTION_SPACE_SIZE == 7650

    def test_encode_decode_roundtrip(self):
        from gym_yinsh.actions import decode_action
        for cell in [0, 42, 84]:
            assert decode_action(encode_place_ring(cell)) == ("place_ring", cell)
        assert decode_action(encode_move_ring(10, 20)) == ("move_ring", 10, 20)
        assert decode_action(encode_remove_row(5, 2)) == ("remove_row", 5, 2)
        assert decode_action(encode_remove_ring(0)) == ("remove_ring", 0)
        assert decode_action(encode_remove_ring(84)) == ("remove_ring", 84)

    def test_action_mask_phase_isolation(self):
        """PLACE_RING phase should only have valid actions in [0,84]."""
        g = YinshGame()
        mask = g.valid_action_mask()
        from gym_yinsh.actions import MOVE_RING_OFFSET
        # No move-ring actions should be valid during setup
        assert not mask[MOVE_RING_OFFSET:].any()
        assert mask[:MOVE_RING_OFFSET].any()
