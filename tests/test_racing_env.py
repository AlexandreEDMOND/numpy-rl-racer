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


def test_progress_reward_increases_with_forward_movement():
    env = RacingEnv()
    env.reset(seed=42)
    obs, reward, done, info = env.step(np.array([0.0, 2.0]))
    assert reward > 0.1, "Progress reward should increase total above base survival"
    assert not done


def test_progress_info_in_info_dict():
    env = RacingEnv()
    env.reset(seed=42)
    obs, reward, done, info = env.step(np.array([0.0, 2.0]))
    assert "progress" in info
    assert isinstance(info["progress"], float)
    assert 0.0 <= info["progress"] <= 1.0


def test_goal_reward_on_lap_completion():
    env = RacingEnv()
    env.reset(seed=42)
    hw, hh = env.track.half_w, env.track.half_h
    # Position the car near the start of the bottom edge (just past the corner)
    env.state.x = np.float64(-hw + 0.2)
    env.state.y = np.float64(-hh + 0.2)
    # Simulate near-completion of a lap to trigger the wrap detection
    env._prev_progress = np.float64(0.99)
    env._prev_seg_idx = 3
    env._cumulative_progress = np.float64(0.99)
    obs, reward, done, info = env.step(np.array([0.0, 2.0]))
    assert reward > 10.0, f"Expected goal_reward (10.0) in reward, got {reward}"
    assert "progress" in info


def test_progress_resets_on_new_episode():
    env = RacingEnv()
    env.reset(seed=42)
    env.step(np.array([0.0, 2.0]))
    env.reset(seed=99)
    obs, reward, done, info = env.step(np.array([0.0, 2.0]))
    assert "progress" in info
    assert 0.0 <= info["progress"] <= 1.0
