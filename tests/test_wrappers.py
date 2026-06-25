import numpy as np
import pytest

from numpy_rl_racer.env.racing_env import Obstacle, RacingEnv
from numpy_rl_racer.env.wrappers import ActionRepeatEnv


def test_skip_frames_validation():
    env = RacingEnv()
    ActionRepeatEnv(env, skip_frames=1)
    with pytest.raises(ValueError, match="skip_frames must be >= 1"):
        ActionRepeatEnv(env, skip_frames=0)
    with pytest.raises(ValueError, match="skip_frames must be >= 1"):
        ActionRepeatEnv(env, skip_frames=-1)


def test_step_calls_inner_env_skip_frames_times():
    call_count = 0

    class MockEnv:
        def step(self, action):
            nonlocal call_count
            call_count += 1
            return np.zeros(6, dtype=np.float64), 1.0, False, {}

        def reset(self, seed=None):
            return np.zeros(6, dtype=np.float64)

    wrapper = ActionRepeatEnv(MockEnv(), skip_frames=4)
    wrapper.step(np.array([0.0, 1.0]))
    assert call_count == 4


def test_reward_accumulation():
    rewards = []

    class MockEnv:
        def step(self, action):
            r = np.float64(len(rewards) + 1)
            rewards.append(r)
            return np.zeros(6, dtype=np.float64), r, False, {}

        def reset(self, seed=None):
            return np.zeros(6, dtype=np.float64)

    wrapper = ActionRepeatEnv(MockEnv(), skip_frames=3)
    _, total_reward, _, _ = wrapper.step(np.array([0.0, 1.0]))
    assert total_reward == np.float64(6.0)


def test_early_termination():
    step_count = 0

    class MockEnv:
        def step(self, action):
            nonlocal step_count
            step_count += 1
            done = step_count >= 2
            return np.zeros(6, dtype=np.float64), 1.0, done, {}

        def reset(self, seed=None):
            return np.zeros(6, dtype=np.float64)

    wrapper = ActionRepeatEnv(MockEnv(), skip_frames=5)
    _, reward, done, _ = wrapper.step(np.array([0.0, 1.0]))
    assert done
    assert step_count == 2
    assert reward == np.float64(2.0)


def test_getattr_passthrough():
    env = RacingEnv()
    wrapper = ActionRepeatEnv(env)
    assert wrapper.track is env.track
    assert wrapper.dt == env.dt
    assert wrapper.env is env


def test_reset_delegation():
    env = RacingEnv()
    wrapper = ActionRepeatEnv(env)
    obs1 = env.reset(seed=42)
    obs2 = wrapper.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


def test_observation_shape_and_dtype():
    env = RacingEnv()
    wrapper = ActionRepeatEnv(env, skip_frames=3)
    wrapper.reset(seed=42)
    obs, _, _, _ = wrapper.step(np.array([0.0, 1.0]))
    assert obs.shape == (6,)
    assert obs.dtype == np.float64


def test_skip_frames_one_is_noop():
    env = RacingEnv()
    wrapper = ActionRepeatEnv(env, skip_frames=1)
    wrapper.reset(seed=42)
    obs1, rew1, done1, info1 = env.step(np.array([0.0, 1.0]))
    wrapper.reset(seed=42)
    obs2, rew2, done2, info2 = wrapper.step(np.array([0.0, 1.0]))
    np.testing.assert_array_equal(obs1, obs2)
    assert rew1 == rew2
    assert done1 == done2
    assert info1 == info2


@pytest.mark.parametrize("track_type", ["rectangular", "circular", "figure8"])
def test_compatible_with_all_track_types(track_type):
    env = RacingEnv(track_type=track_type)
    wrapper = ActionRepeatEnv(env, skip_frames=2)
    wrapper.reset(seed=42)
    obs, reward, done, info = wrapper.step(np.array([0.0, 1.0]))
    assert obs.shape == (6,)
    assert obs.dtype == np.float64
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_compatible_with_obstacles():
    obstacles = [Obstacle(0.0, 0.0, 0.5)]
    env = RacingEnv(obstacles=obstacles)
    wrapper = ActionRepeatEnv(env, skip_frames=2)
    wrapper.reset(seed=42)
    obs, reward, done, info = wrapper.step(np.array([0.0, 1.0]))
    assert obs.shape == (8,)
    assert obs.dtype == np.float64
