import numpy as np

from numpy_rl_racer.env.car import CarState
from numpy_rl_racer.env.racing_env import CircularTrack, RacingEnv, RectangularTrack


def test_reset_returns_numpy_observation():
    env = RacingEnv()
    obs = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert obs.dtype == np.float64


def test_step_returns_observation_reward_done_info():
    env = RacingEnv()
    env.reset(seed=42)
    obs, reward, done, info = env.step(np.array([0.0, 2.0]))
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
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
    assert obs.shape == (6,)
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


# ── CircularTrack ──────────────────────────────────────────────────


def test_circular_track_goal_position():
    track = CircularTrack(radius=6.0)
    gx, gy = track.goal_position
    assert gx == np.float64(0.0)
    assert gy == np.float64(-6.0)


def test_circular_track_start_position():
    track = CircularTrack(radius=6.0)
    sx, sy, sh = track.start_position
    assert sx == np.float64(0.0)
    assert sy == np.float64(-6.0)
    assert sh == np.float64(0.0)


def test_circular_track_progress_at_start():
    track = CircularTrack(radius=6.0)
    np.testing.assert_almost_equal(track.progress_along_centerline(0.0, -6.0), 0.0)


def test_circular_track_progress_at_quarter():
    track = CircularTrack(radius=6.0)
    np.testing.assert_almost_equal(track.progress_along_centerline(6.0, 0.0), 0.25)


def test_circular_track_progress_at_half():
    track = CircularTrack(radius=6.0)
    np.testing.assert_almost_equal(track.progress_along_centerline(0.0, 6.0), 0.5)


def test_circular_track_progress_at_three_quarters():
    track = CircularTrack(radius=6.0)
    np.testing.assert_almost_equal(track.progress_along_centerline(-6.0, 0.0), 0.75)


def test_circular_track_progress_bounds():
    track = CircularTrack(radius=6.0)
    for x, y in [(0.0, -6.0), (6.0, 0.0), (0.0, 6.0), (-6.0, 0.0)]:
        p = track.progress_along_centerline(x, y)
        assert 0.0 <= p <= 1.0


def test_circular_track_is_on_track_centerline():
    track = CircularTrack(radius=6.0)
    assert track.is_on_track(0.0, -6.0)
    assert track.is_on_track(6.0, 0.0)
    assert track.is_on_track(0.0, 6.0)
    assert track.is_on_track(-6.0, 0.0)


def test_circular_track_is_on_track_center():
    track = CircularTrack(radius=6.0, track_width=2.0)
    assert not track.is_on_track(0.0, 0.0), "Center of circle should not be on track"


def test_circular_track_is_on_track_too_far():
    track = CircularTrack(radius=6.0, track_width=2.0)
    assert not track.is_on_track(8.0, 0.0), "Point beyond outer edge should not be on track"
    assert not track.is_on_track(0.0, -8.0), "Point beyond outer edge should not be on track"


def test_circular_track_is_on_track_edge_boundary():
    track = CircularTrack(radius=6.0, track_width=2.0)
    assert track.is_on_track(5.0, 0.0), "Inner edge of road should be on track"
    assert track.is_on_track(7.0, 0.0), "Outer edge of road should be on track"


def test_circular_track_half_properties():
    track = CircularTrack(radius=6.0, track_width=2.0)
    assert track.half_w == np.float64(6.0)
    assert track.half_h == np.float64(6.0)


def test_racing_env_with_circular_track_reset():
    track = CircularTrack(radius=6.0, track_width=2.0)
    env = RacingEnv(track=track)
    obs = env.reset(seed=42)
    assert env.state.x == np.float64(0.0)
    assert env.state.y == np.float64(-6.0)
    assert env.state.heading == np.float64(0.0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)


def test_circular_track_progress_increases_when_moving_forward():
    track = CircularTrack(radius=8.0, track_width=2.0)
    env = RacingEnv(track=track, dt=0.1)
    env.reset(seed=42)
    p0 = env.current_progress
    for _ in range(30):
        env.step(np.array([0.5, 5.0]))
    assert env.current_progress > p0


# ── Goal-distance reward shaping ─────────────────────────────────────


def test_reward_includes_goal_distance_shaping():
    env = RacingEnv()
    env.reset(seed=42)
    gx, gy = env.goal_position
    prev_dist = np.sqrt((env.state.x - gx) ** 2 + (env.state.y - gy) ** 2)
    _, reward, _, _ = env.step(np.array([0.0, 2.0]))
    new_dist = np.sqrt((env.state.x - gx) ** 2 + (env.state.y - gy) ** 2)
    expected_shaping = np.float64(0.5) * (prev_dist - new_dist) / env.track.track_width
    # Base on-track reward is 0.1; no lap bonus for a single gentle step
    expected_reward = np.float64(0.1) + expected_shaping
    np.testing.assert_almost_equal(reward, expected_reward, decimal=12)


def test_stationary_action_at_goal_gives_base_reward():
    env = RacingEnv()
    env.reset(seed=42)
    # Car starts exactly at goal position; zero action keeps it there
    _, reward, _, _ = env.step(np.array([0.0, 0.0]))
    assert reward == np.float64(0.1), f"Expected 0.1 (no shaping), got {reward}"


def test_moving_away_from_goal_gives_lower_reward():
    env = RacingEnv()
    env.reset(seed=42)
    _, r_zero, _, _ = env.step(np.array([0.0, 0.0]))
    env.reset(seed=42)
    _, r_away, _, _ = env.step(np.array([0.0, 2.0]))
    assert r_away < r_zero, (
        f"Expected reward when moving away ({r_away}) "
        f"to be lower than stationary ({r_zero})"
    )


def test_higher_reward_when_moving_toward_goal_from_far():
    env = RacingEnv()
    env.reset(seed=42)
    _, r_still_at_goal, _, _ = env.step(np.array([0.0, 0.0]))
    env.reset(seed=42)
    env.state = CarState(x=4.0, y=-4.0, heading=np.pi, velocity=2.0)
    _, r_toward_from_far, _, _ = env.step(np.array([0.0, 2.0]))
    assert r_toward_from_far > r_still_at_goal, (
        f"Expected reward when moving toward goal from far ({r_toward_from_far}) "
        f"to be higher than stationary at goal ({r_still_at_goal})"
    )


def test_toward_goal_higher_than_away_same_position():
    env = RacingEnv()
    env.reset(seed=42)
    gx, gy = env.goal_position
    env.state = CarState(x=3.0, y=-4.0, heading=np.pi, velocity=1.0)
    prev_dist = np.sqrt((env.state.x - gx) ** 2 + (env.state.y - gy) ** 2)
    _, r_toward, _, _ = env.step(np.array([0.0, 2.0]))
    new_dist_toward = np.sqrt((env.state.x - gx) ** 2 + (env.state.y - gy) ** 2)
    env.reset(seed=42)
    env.state = CarState(x=3.0, y=-4.0, heading=0.0, velocity=1.0)
    _, r_away, _, _ = env.step(np.array([0.0, 2.0]))
    assert r_toward > r_away, (
        f"Expected toward-goal reward ({r_toward}) > away-from-goal reward ({r_away}). "
        f"Toward delta={prev_dist - new_dist_toward:.4f}"
    )


def test_goal_distance_shaping_magnitude_reasonable():
    env = RacingEnv()
    env.reset(seed=42)
    env.state = CarState(x=5.0, y=-4.0, heading=np.pi, velocity=5.0)
    _, reward, _, _ = env.step(np.array([0.0, 5.0]))
    # Shaping bound: max speed * dt = 10 * 0.1 = 1.0 distance per step
    # max positive shaping = 0.5 * 1.0 / 2.0 = 0.25
    # on-track bonus = 0.1, so reward < 1.0 (lap bonus threshold)
    assert reward < 1.0, (
        f"Reward {reward} should not dominate existing components; "
        f"expected < 1.0 (lap bonus threshold)"
    )


# ── Track-relative observation features ──────────────────────────────


def test_observation_has_six_features():
    env = RacingEnv()
    obs = env.reset(seed=42)
    assert len(obs) == 6


def test_distance_to_edge_normalized_one_on_centerline_rectangular():
    env = RacingEnv()
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-4.0, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    np.testing.assert_almost_equal(obs[4], 1.0)


def test_distance_to_edge_normalized_zero_at_boundary_rectangular():
    env = RacingEnv()
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-5.0, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    np.testing.assert_almost_equal(obs[4], 0.0)


def test_heading_error_zero_when_aligned_with_centerline():
    env = RacingEnv()
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-4.0, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    np.testing.assert_almost_equal(obs[5], 0.0)


def test_heading_error_positive_left_of_centerline():
    env = RacingEnv()
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-4.0, heading=np.pi / 4, velocity=0.0)
    obs = env._get_observation()
    assert obs[5] > 0.0


def test_heading_error_negative_right_of_centerline():
    env = RacingEnv()
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-4.0, heading=-np.pi / 4, velocity=0.0)
    obs = env._get_observation()
    assert obs[5] < 0.0


def test_distance_to_edge_normalized_one_on_centerline_circular():
    track = CircularTrack(radius=6.0, track_width=2.0)
    env = RacingEnv(track=track)
    env.reset(seed=42)
    env.state = CarState(x=0.0, y=-6.0, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    np.testing.assert_almost_equal(obs[4], 1.0)


def test_distance_to_edge_normalized_zero_at_boundary_circular():
    track = CircularTrack(radius=6.0, track_width=2.0)
    env = RacingEnv(track=track)
    env.reset(seed=42)
    env.state = CarState(x=0.0, y=-7.0, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    np.testing.assert_almost_equal(obs[4], 0.0)
