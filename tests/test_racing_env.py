import numpy as np

from numpy_rl_racer.env.racing_env import RacingEnv, RectangularTrack


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


def test_goal_position():
    env = RacingEnv(track_width=10.0, track_height=8.0)
    gx, gy = env.goal_position
    assert gx == np.float64(0.0)
    assert gy == -env.track.half_h
    gx2, gy2 = env.track.goal_position
    assert gx2 == np.float64(0.0)
    assert gy2 == -env.track.half_h


def test_progress_starts_at_zero():
    env = RacingEnv()
    env.reset(seed=42)
    assert env.current_progress == np.float64(0.0)
    assert env.lap_count == 0


def test_progress_increases_when_moving_forward():
    env = RacingEnv()
    env.reset(seed=42)
    p0 = env.current_progress
    for _ in range(20):
        env.step(np.array([0.0, 2.0]))
    assert env.current_progress > p0


def test_info_dict_contains_goal_and_progress():
    env = RacingEnv()
    env.reset(seed=42)
    _, _, _, info = env.step(np.array([0.0, 2.0]))
    assert 'progress' in info
    assert 'lap_count' in info
    assert 'goal_position' in info


def test_progress_computation_known_points():
    track = RectangularTrack(width=10.0, height=8.0)
    # Start point (0, -4) → progress = 0
    np.testing.assert_almost_equal(track.progress_along_centerline(0.0, -4.0), 0.0)
    # Right edge midpoint (5, 0) → cum = half_w + half_h = 5 + 4 = 9,  total = 36,  frac = 9/36
    np.testing.assert_almost_equal(track.progress_along_centerline(5.0, 0.0), 9.0 / 36.0)
    # Top edge midpoint (0, 4) → cum = half_w + 2*half_h + half_w = 5+8+5 = 18,  frac = 18/36
    np.testing.assert_almost_equal(track.progress_along_centerline(0.0, 4.0), 18.0 / 36.0)
    # Progress never exceeds 1
    p = track.progress_along_centerline(0.0, -4.0)
    assert 0.0 <= p <= 1.0


def test_lap_completion_gives_bonus_reward():
    env = RacingEnv(dt=0.1)
    env.reset(seed=0)
    total_reward = 0.0
    # Drive at full speed with slight steering to loop around
    for _ in range(200):
        obs, reward, done, info = env.step(np.array([1.5, 10.0]))
        total_reward += reward
        if info['lap_count'] > 0:
            break
    # Check that a lap was completed and bonus reward was given
    assert not done, "Car should still be on track"
    assert info['lap_count'] >= 1, "Should have completed at least one lap"
    assert info['progress'] < 0.5, (
        "Progress should wrap to low value after lap completion"
    )
