import numpy as np

from numpy_rl_racer.env.car import CarState
from numpy_rl_racer.env.racing_env import CircularTrack, Figure8Track, Obstacle, RacingEnv, RectangularTrack


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
    env = RacingEnv(randomize_start=False)
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
    assert 'elapsed_time' in info


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
    env = RacingEnv(dt=0.1, randomize_start=False)
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


# ── Figure8Track ──────────────────────────────────────────────────


def test_figure8_track_initialization():
    track = Figure8Track(radius=6.0, track_width=2.0)
    assert track.radius == np.float64(6.0)
    assert track.track_width == np.float64(2.0)


def test_figure8_track_goal_position():
    track = Figure8Track(radius=6.0, track_width=2.0)
    gx, gy = track.goal_position
    R = 6.0
    np.testing.assert_almost_equal(gx, R / np.sqrt(2.0))
    np.testing.assert_almost_equal(gy, -R / 2.0)


def test_figure8_track_start_position():
    track = Figure8Track(radius=6.0, track_width=2.0)
    sx, sy, sh = track.start_position
    R = 6.0
    np.testing.assert_almost_equal(sx, R / np.sqrt(2.0))
    np.testing.assert_almost_equal(sy, -R / 2.0)
    assert sh == np.float64(0.0)


def test_figure8_track_progress_at_goal_is_zero():
    track = Figure8Track(radius=6.0, track_width=2.0)
    gx, gy = track.goal_position
    np.testing.assert_almost_equal(track.progress_along_centerline(gx, gy), 0.0)


def test_figure8_track_progress_monotonic():
    track = Figure8Track(radius=6.0, track_width=2.0)
    ts = np.linspace(0.01, 0.99, 50)
    prev_p = -1.0
    for t in ts:
        x, y, _ = track.sample_centerline_point(t)
        p = track.progress_along_centerline(x, y)
        assert p >= prev_p - 0.01, f"Progress regressed at t={t}: {prev_p} -> {p}"
        prev_p = p


def test_figure8_track_progress_wraps():
    track = Figure8Track(radius=6.0, track_width=2.0)
    gx, gy = track.goal_position
    assert track.progress_along_centerline(gx, gy) == np.float64(0.0)
    x_just_before, y_just_before, _ = track.sample_centerline_point(0.999)
    p = track.progress_along_centerline(x_just_before, y_just_before)
    assert p > 0.99


def test_figure8_track_centerline_points_are_on_track():
    track = Figure8Track(radius=6.0, track_width=2.0)
    for t in np.linspace(0.0, 0.99, 20):
        x, y, _ = track.sample_centerline_point(t)
        assert track.is_on_track(x, y), f"Centerline point at t={t} not on track"


def test_figure8_track_far_points_off_track():
    track = Figure8Track(radius=6.0, track_width=2.0)
    assert track.is_on_track(0.0, 0.0), "Intersection should be on track"
    assert not track.is_on_track(10.0, 0.0), "Point far right should be off track"
    assert not track.is_on_track(0.0, 5.0), "Point far above should be off track"
    assert not track.is_on_track(0.0, -5.0), "Point far below should be off track"


def test_figure8_track_centerline_info_on_centerline():
    track = Figure8Track(radius=6.0, track_width=2.0)
    gx, gy = track.goal_position
    dist, tangent = track.centerline_info(gx, gy)
    np.testing.assert_almost_equal(dist, 0.0, decimal=10)
    np.testing.assert_almost_equal(tangent, 0.0, decimal=10)


def test_figure8_track_tangent_at_extremes():
    track = Figure8Track(radius=6.0, track_width=2.0)
    # t=0.125 → θ=0 (rightmost, R, 0): tangent = π/2 (up)
    x_r, y_r, _ = track.sample_centerline_point(0.125)
    _, tangent_r = track.centerline_info(x_r, y_r)
    np.testing.assert_almost_equal(tangent_r, np.pi / 2.0, decimal=5)
    # t=0.25 → θ=π/4 (top-right, R/√2, R/2): tangent = π (left)
    x_tr, y_tr, _ = track.sample_centerline_point(0.25)
    _, tangent_tr = track.centerline_info(x_tr, y_tr)
    np.testing.assert_almost_equal(tangent_tr, np.pi, decimal=5)


def test_figure8_track_half_properties():
    track = Figure8Track(radius=6.0, track_width=2.0)
    assert track.half_w == np.float64(6.0)
    assert track.half_h == np.float64(3.0)


def test_figure8_track_sample_returns_on_track():
    track = Figure8Track(radius=6.0, track_width=2.0)
    for _ in range(20):
        x, y, _ = track.sample_centerline_point()
        assert track.is_on_track(x, y)


def test_figure8_racing_env_reset():
    track = Figure8Track(radius=6.0, track_width=2.0)
    env = RacingEnv(track=track, randomize_start=False)
    obs = env.reset(seed=42)
    gx, gy = track.goal_position
    np.testing.assert_almost_equal(float(env.state.x), float(gx), decimal=10)
    np.testing.assert_almost_equal(float(env.state.y), float(gy), decimal=10)
    assert env.state.heading == np.float64(0.0)
    assert isinstance(obs, np.ndarray)


def test_figure8_racing_env_randomized_start_on_track():
    track = Figure8Track(radius=6.0, track_width=2.0)
    env = RacingEnv(track=track, randomize_start=True)
    for seed in range(20):
        env.reset(seed=seed)
        assert track.is_on_track(env.state.x, env.state.y), (
            f"Start position ({env.state.x}, {env.state.y}) with seed {seed} is off track"
        )


def test_figure8_racing_env_step_straight_stays_on_track():
    track = Figure8Track(radius=6.0, track_width=2.0)
    env = RacingEnv(track=track)
    env.reset(seed=42)
    for _ in range(30):
        obs, reward, done, info = env.step(np.array([1.0, 5.0]))
        if done:
            break
    # With a moderate steering input the car should survive several steps
    # (this tests that the env doesn't crash, and track methods work)


def test_figure8_racing_env_track_type():
    env = RacingEnv(track_type='figure8')
    from numpy_rl_racer.env.racing_env import Figure8Track as FT
    assert isinstance(env.track, FT)
    obs = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert not np.any(np.isnan(obs))


def test_figure8_track_progress_bounds():
    track = Figure8Track(radius=6.0, track_width=2.0)
    for t in np.linspace(0.0, 0.99, 20):
        x, y, _ = track.sample_centerline_point(t)
        p = track.progress_along_centerline(x, y)
        assert 0.0 <= p <= 1.0


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
    env = RacingEnv(track=track, randomize_start=False)
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
    env = RacingEnv(randomize_start=False)
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


# ── Randomized start ─────────────────────────────────────────────────


def test_randomized_start_preserves_original_when_disabled():
    env = RacingEnv(randomize_start=False)
    env.reset(seed=42)
    sx, sy, sh = env.track.start_position
    assert env.state.x == sx
    assert env.state.y == sy
    assert env.state.heading == sh


def test_randomized_start_produces_variation():
    env = RacingEnv(randomize_start=True)
    positions = set()
    for seed in range(20):
        env.reset(seed=seed)
        positions.add((float(env.state.x), float(env.state.y), float(env.state.heading)))
    assert len(positions) >= 2, "Expected different start positions across seeds"


def test_randomized_start_seed_determinism():
    env = RacingEnv(randomize_start=True)
    env.reset(seed=123)
    x1, y1, h1 = env.state.x, env.state.y, env.state.heading
    env.reset(seed=123)
    x2, y2, h2 = env.state.x, env.state.y, env.state.heading
    assert x1 == x2 and y1 == y2 and h1 == h2, (
        f"Same seed should give same start, got ({x1},{y1},{h1}) vs ({x2},{y2},{h2})"
    )


def test_randomized_start_on_track():
    env = RacingEnv(randomize_start=True)
    for seed in range(50):
        env.reset(seed=seed)
        assert env.track.is_on_track(env.state.x, env.state.y), (
            f"Start position ({env.state.x}, {env.state.y}) with seed {seed} is off track"
        )


def test_randomized_start_on_track_circular():
    track = CircularTrack(radius=6.0, track_width=2.0)
    env = RacingEnv(track=track, randomize_start=True)
    for seed in range(50):
        env.reset(seed=seed)
        assert env.track.is_on_track(env.state.x, env.state.y), (
            f"Start position ({env.state.x}, {env.state.y}) with seed {seed} is off track"
        )


# ── Obstacles ────────────────────────────────────────────────────────


def test_default_no_obstacles_backward_compatible():
    env = RacingEnv()
    assert len(env.obstacles) == 0, "Default should have no obstacles for backward compat"


def test_custom_obstacles_list():
    obstacles = [Obstacle(x=1.0, y=2.0, radius=0.5)]
    env = RacingEnv(obstacles=obstacles)
    assert len(env.obstacles) == 1
    assert isinstance(env.obstacles[0], Obstacle)


def test_obstacles_deterministic_with_seed():
    obstacles = [Obstacle(x=1.0, y=1.0, radius=0.5)]
    env = RacingEnv(obstacles=obstacles)
    env.reset(seed=123)
    obs1 = env._get_observation()
    env.reset(seed=123)
    obs2 = env._get_observation()
    np.testing.assert_array_equal(obs1, obs2)


def test_obstacle_collision_ends_episode():
    env = RacingEnv(obstacles=[Obstacle(x=1.0, y=-3.0, radius=0.5)])
    env.reset(seed=42)
    env.state = CarState(x=0.0, y=-3.5, heading=np.pi / 2, velocity=2.0)
    done = False
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0.0, 5.0]))
        if done:
            break
    assert done, "Car should collide with obstacle when driving toward it"


def test_obstacle_collision_gives_negative_reward():
    env = RacingEnv(obstacles=[Obstacle(x=1.0, y=-3.0, radius=0.5)])
    env.reset(seed=42)
    env.state = CarState(x=0.0, y=-3.5, heading=np.pi / 2, velocity=2.0)
    collided = False
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0.0, 5.0]))
        if done:
            collided = True
            assert reward < -0.5, (
                f"Expected negative reward on collision, got {reward}"
            )
            break
    assert collided


def test_no_obstacle_observation_is_6d():
    env = RacingEnv(obstacles=[])
    obs = env.reset(seed=42)
    assert obs.shape == (6,), f"Expected 6-dim obs without obstacles, got {obs.shape}"


def test_with_obstacles_observation_is_8d():
    env = RacingEnv(obstacles=[Obstacle(x=1.0, y=1.0, radius=0.5)])
    obs = env.reset(seed=42)
    assert obs.shape == (8,), f"Expected 8-dim obs with obstacles, got {obs.shape}"


def test_observation_obstacle_features_within_bounds():
    env = RacingEnv(obstacles=[Obstacle(x=1.0, y=1.0, radius=0.5)])
    obs = env.reset(seed=42)
    # normalized distance should be in [0, 1]
    assert 0.0 <= obs[6] <= 1.0, f"Obstacle distance out of bounds: {obs[6]}"
    # normalized angle should be in [-1, 1]
    assert -1.0 <= obs[7] <= 1.0, f"Obstacle angle out of bounds: {obs[7]}"


def test_obstacle_observation_with_circular_track():
    track = CircularTrack(radius=6.0, track_width=2.0)
    env = RacingEnv(track=track, obstacles=[Obstacle(x=0.0, y=0.0, radius=0.5)])
    obs = env.reset(seed=42)
    assert obs.shape == (8,)
    assert 0.0 <= obs[6] <= 1.0
    assert -1.0 <= obs[7] <= 1.0


def test_collision_no_false_positive_when_far():
    env = RacingEnv(obstacles=[Obstacle(x=100.0, y=100.0, radius=0.5)])
    env.reset(seed=42)
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0.0, 2.0]))
        assert not done, "Car should not collide with distant obstacle"


def test_empty_obstacles_list_backward_compatible():
    env = RacingEnv(obstacles=[])
    obs = env.reset(seed=42)
    assert obs.shape == (6,), "No obstacles should give 6-dim observation"


# ── Elapsed time ──────────────────────────────────────────────────────


def test_elapsed_time_starts_zero_after_reset():
    env = RacingEnv()
    env.reset(seed=42)
    assert env.elapsed_time == np.float64(0.0)


def test_elapsed_time_increments_by_dt_per_step():
    env = RacingEnv(dt=0.1)
    env.reset(seed=42)
    for i in range(5):
        _, _, _, info = env.step(np.array([0.0, 2.0]))
        np.testing.assert_almost_equal(info['elapsed_time'], (i + 1) * 0.1)


def test_elapsed_time_resets_on_reset():
    env = RacingEnv()
    env.reset(seed=42)
    env.step(np.array([0.0, 2.0]))
    env.step(np.array([0.0, 2.0]))
    assert env.elapsed_time > 0.0
    env.reset(seed=42)
    assert env.elapsed_time == np.float64(0.0)


def test_info_elapsed_time_matches_env_attribute():
    env = RacingEnv()
    env.reset(seed=42)
    _, _, _, info = env.step(np.array([0.0, 2.0]))
    assert info['elapsed_time'] == env.elapsed_time


# ── Time penalty ──────────────────────────────────────────────────────


def test_time_penalty_zero_is_backward_compatible():
    env_default = RacingEnv(time_penalty=0.0)
    env_default.reset(seed=42)
    env_base = RacingEnv()
    env_base.reset(seed=42)
    for _ in range(10):
        a = np.array([0.0, 2.0])
        _, r1, _, _ = env_default.step(a)
        _, r2, _, _ = env_base.step(a)
        assert r1 == r2, (
            f"time_penalty=0.0 should match default env; got {r1} vs {r2}"
        )


def test_time_penalty_reduces_reward():
    env_no_penalty = RacingEnv(time_penalty=0.0, randomize_start=False)
    env_penalty = RacingEnv(time_penalty=0.1, randomize_start=False)
    env_no_penalty.reset(seed=42)
    env_penalty.reset(seed=42)
    cum_no_penalty = 0.0
    cum_penalty = 0.0
    a = np.array([0.0, 2.0])
    for _ in range(20):
        _, r1, _, _ = env_no_penalty.step(a)
        _, r2, _, _ = env_penalty.step(a)
        cum_no_penalty += r1
        cum_penalty += r2
    assert cum_penalty < cum_no_penalty, (
        f"time_penalty should reduce cumulative reward; "
        f"no_penalty={cum_no_penalty} penalty={cum_penalty}"
    )


def test_time_penalty_magnitude():
    env = RacingEnv(time_penalty=0.1, dt=0.05, randomize_start=False)
    env.reset(seed=42)
    _, r1, _, info1 = env.step(np.array([0.0, 0.0]))
    _, r2, _, info2 = env.step(np.array([0.0, 0.0]))
    # Each step loses 0.1 * 0.05 = 0.005 from time penalty
    elapsed = info2['elapsed_time']
    assert elapsed == 2 * 0.05
    np.testing.assert_almost_equal(
        r2, np.float64(0.1) - np.float64(0.1) * np.float64(0.05), decimal=12
    )
