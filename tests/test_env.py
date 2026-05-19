"""Tests for the Gymnasium environment."""
import numpy as np
import pytest
import gymnasium as gym
import gym_yinsh


@pytest.fixture
def env():
    e = gym.make("yinsh-v0")
    yield e
    e.close()


class TestGymAPI:
    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=0)
        assert "board" in obs
        assert "context" in obs
        assert obs["board"].shape == (4, 85)
        assert obs["context"].shape == (8,)

    def test_obs_in_observation_space(self, env):
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)

    def test_action_space_size(self, env):
        assert env.action_space.n == 7650

    def test_step_valid_action(self, env):
        obs, info = env.reset(seed=0)
        mask = info["action_mask"]
        valid = np.where(mask)[0]
        obs2, reward, term, trunc, info2 = env.step(valid[0])
        assert env.observation_space.contains(obs2)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)

    def test_step_invalid_action_returns_penalty(self, env):
        obs, info = env.reset(seed=0)
        mask = info["action_mask"]
        invalid = np.where(~mask)[0]
        if len(invalid) == 0:
            pytest.skip("No invalid actions")
        _, reward, term, _, _ = env.step(invalid[0])
        assert reward == -1.0
        assert term

    def test_info_contains_action_mask(self, env):
        _, info = env.reset()
        assert "action_mask" in info
        assert info["action_mask"].shape == (7650,)
        assert info["action_mask"].dtype == bool

    def test_action_masks_method(self, env):
        env.reset()
        mask = env.unwrapped.action_masks()
        assert mask.shape == (7650,)
        assert mask.dtype == bool

    def test_full_episode(self, env):
        obs, info = env.reset(seed=123)
        rng = np.random.default_rng(123)
        for _ in range(2000):
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            assert len(valid) > 0
            action = rng.choice(valid)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                assert info["winner"] in (0, 1)
                return
        pytest.fail("Episode did not terminate within 2000 steps")

    def test_render_ansi(self):
        e = gym.make("yinsh-v0", render_mode="ansi")
        e.reset()
        result = e.render()
        assert isinstance(result, str)
        assert "Phase:" in result
        e.close()

    def test_board_channels_binary(self, env):
        obs, _ = env.reset()
        board = obs["board"]
        assert board.min() >= 0
        assert board.max() <= 1

    def test_context_range(self, env):
        obs, _ = env.reset()
        ctx = obs["context"]
        hi = np.array([4, 1, 3, 3, 5, 5, 5, 5])
        assert (ctx >= 0).all()
        assert (ctx <= hi).all()
