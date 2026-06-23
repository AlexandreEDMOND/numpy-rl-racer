import numpy as np

from numpy_rl_racer.env.racing_env import RacingEnv


def test_reset_returns_numpy_observation():
    env = RacingEnv()
    obs = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert obs.dtype == np.float64


def test_step_returns_observation_reward_done_info():
    env = RacingEnv()
    env.reset(seed=42)
    obs, reward, done, info = env.step(np.array([0.0, 2.0]))
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_leaving_track_ends_episode():
    env = RacingEnv()
    env.reset(seed=42)
    done = False
    for _ in range(50):
        obs, reward, d, info = env.step(np.array([5.0, 10.0]))
        if d:
            done = True
            break
    assert done, "Car should eventually leave the track when steering hard"


def test_staying_on_track_does_not_end_episode():
    env = RacingEnv()
    env.reset(seed=42)
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0.0, 2.0]))
        assert not done, "Car should stay on track with gentle forward motion"
        assert reward > 0.0, "Reward should be positive for staying on track"


def test_respects_seed_determinism():
    env1 = RacingEnv()
    env2 = RacingEnv()
    obs1 = env1.reset(seed=123)
    obs2 = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)


def test_observation_contains_x_y_heading_velocity():
    env = RacingEnv()
    obs = env.reset(seed=42)
    assert obs.shape == (4,)
    assert not np.any(np.isnan(obs))
