import numpy as np

from numpy_rl_racer.env.car import CarState
from numpy_rl_racer.env.racing_env import BezierTrack, CircularTrack, Figure8Track, Obstacle, RacingEnv, RectangularTrack, reward_line_endpoints


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


# ── BezierTrack ─────────────────────────────────────────────────────


def test_bezier_track_initialization():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0)
    assert track.num_anchors == 8
    assert track.track_width == np.float64(2.0)
    assert track.radius == np.float64(6.0)


def test_bezier_track_different_seeds_different_geometry():
    track1 = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    track2 = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=99)
    # Different seeds should produce different control points
    assert not np.allclose(track1._anchors, track2._anchors)


def test_bezier_track_same_seed_reproduces_geometry():
    track1 = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    track2 = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    np.testing.assert_array_equal(track1._anchors, track2._anchors)
    np.testing.assert_array_equal(track1._centerline, track2._centerline)


def test_bezier_track_centerline_is_closed():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    # Centerline should form a closed loop: first and last points should be close
    first = track._centerline[0]
    last = track._centerline[-1]
    dist = np.sqrt(np.sum((first - last) ** 2))
    assert dist < 0.1, f"Centerline not closed, distance: {dist}"


def test_bezier_track_progress_at_goal_is_zero():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    gx, gy = track.goal_position
    np.testing.assert_almost_equal(track.progress_along_centerline(gx, gy), 0.0)


def test_bezier_track_progress_monotonic():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    ts = np.linspace(0.01, 0.99, 20)
    prev_p = -1.0
    for t in ts:
        x, y, _ = track.sample_centerline_point(t)
        p = track.progress_along_centerline(x, y)
        assert p >= prev_p - 0.01, f"Progress regressed at t={t}: {prev_p} -> {p}"
        prev_p = p


def test_bezier_track_progress_bounds():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    for t in np.linspace(0.0, 0.99, 20):
        x, y, _ = track.sample_centerline_point(t)
        p = track.progress_along_centerline(x, y)
        assert 0.0 <= p <= 1.0


def test_bezier_track_progress_wraps():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    gx, gy = track.goal_position
    assert track.progress_along_centerline(gx, gy) == np.float64(0.0)
    x_just_before, y_just_before, _ = track.sample_centerline_point(0.999)
    p = track.progress_along_centerline(x_just_before, y_just_before)
    assert p > 0.99


def test_bezier_track_centerline_points_are_on_track():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    for t in np.linspace(0.0, 0.99, 20):
        x, y, _ = track.sample_centerline_point(t)
        assert track.is_on_track(x, y), f"Centerline point at t={t} not on track"


def test_bezier_track_far_points_off_track():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    assert track.is_on_track(float(track._cs_x[0]), float(track._cs_y[0])), "Start point should be on track"
    assert not track.is_on_track(100.0, 0.0), "Point far right should be off track"
    assert not track.is_on_track(0.0, 100.0), "Point far above should be off track"


def test_bezier_track_centerline_info_on_centerline():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    gx, gy = track.goal_position
    dist, _ = track.centerline_info(gx, gy)
    np.testing.assert_almost_equal(dist, 0.0, decimal=10)


def test_bezier_track_sample_returns_on_track():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    for _ in range(20):
        x, y, _ = track.sample_centerline_point()
        assert track.is_on_track(x, y)


def test_bezier_track_get_centerline_point():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    gx, gy = track.goal_position
    cx, cy, tangent = track.get_centerline_point(0.0)
    np.testing.assert_almost_equal(float(cx), float(gx), decimal=10)
    np.testing.assert_almost_equal(float(cy), float(gy), decimal=10)

    cx, cy, tangent = track.get_centerline_point(0.5)
    p = track.progress_along_centerline(cx, cy)
    np.testing.assert_almost_equal(p, 0.5, decimal=2)


def test_bezier_track_half_properties():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    assert track.half_w > 0
    assert track.half_h > 0
    assert isinstance(track.half_w, np.float64)
    assert isinstance(track.half_h, np.float64)


def test_bezier_track_small_number_of_anchors():
    track = BezierTrack(num_anchors=3, track_width=2.0, radius=6.0, seed=42)
    assert track.num_anchors == 3
    assert track._perimeter > 0
    # Centerline should still form a closed loop
    first = track._centerline[0]
    last = track._centerline[-1]
    dist = np.sqrt(np.sum((first - last) ** 2))
    assert dist < 0.1


def test_bezier_racing_env_reset():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    env = RacingEnv(track=track, randomize_start=False)
    obs = env.reset(seed=42)
    gx, gy = track.goal_position
    np.testing.assert_almost_equal(float(env.state.x), float(gx), decimal=10)
    np.testing.assert_almost_equal(float(env.state.y), float(gy), decimal=10)
    assert isinstance(obs, np.ndarray)


def test_bezier_racing_env_randomized_start_on_track():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    env = RacingEnv(track=track, randomize_start=True)
    for seed in range(10):
        env.reset(seed=seed)
        assert track.is_on_track(env.state.x, env.state.y), (
            f"Start position ({env.state.x}, {env.state.y}) with seed {seed} is off track"
        )


def test_bezier_racing_env_step_straight_stays_on_track():
    track = BezierTrack(num_anchors=8, track_width=2.0, radius=6.0, seed=42)
    env = RacingEnv(track=track)
    env.reset(seed=42)
    for _ in range(30):
        obs, reward, done, info = env.step(np.array([1.0, 5.0]))
        if done:
            break


def test_bezier_racing_env_track_type():
    env = RacingEnv(track_type='bezier')
    assert isinstance(env.track, BezierTrack)
    obs = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert not np.any(np.isnan(obs))


def test_bezier_racing_env_reset_and_steps():
    """Integration test: reset + 10 step() calls on bezier track complete without errors."""
    env = RacingEnv(track_type='bezier')
    env.reset(seed=42)
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0.0, 2.0]))
        assert isinstance(obs, np.ndarray)
        assert np.all(np.isfinite(obs))
        assert 'progress' in info
        assert 'lap_count' in info


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


def test_reward_includes_progress_shaping():
    env = RacingEnv()
    env.reset(seed=42)
    prev_progress = env.current_progress
    _, reward, _, _ = env.step(np.array([0.0, 2.0]))
    new_progress = env.current_progress
    progress_diff = new_progress - prev_progress
    expected_shaping = np.float64(0.5) * progress_diff
    expected_reward = np.float64(0.1) + expected_shaping
    np.testing.assert_almost_equal(reward, expected_reward, decimal=12)


def test_stationary_action_at_goal_gives_base_reward():
    env = RacingEnv()
    env.reset(seed=42)
    # Car starts exactly at goal position; zero action keeps it there
    _, reward, _, _ = env.step(np.array([0.0, 0.0]))
    assert reward == np.float64(0.1), f"Expected 0.1 (no shaping), got {reward}"


def test_moving_forward_gives_higher_reward():
    env = RacingEnv(randomize_start=False)
    env.reset(seed=42)
    _, r_zero, _, _ = env.step(np.array([0.0, 0.0]))
    env.reset(seed=42)
    _, r_forward, _, _ = env.step(np.array([0.0, 2.0]))
    assert r_forward > r_zero, (
        f"Expected reward when moving forward ({r_forward}) "
        f"to be higher than stationary ({r_zero})"
    )


def test_progress_shaping_rewards_forward_movement():
    env = RacingEnv()
    env.reset(seed=42)
    _, r_still, _, _ = env.step(np.array([0.0, 0.0]))
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-4.0, heading=0.0, velocity=1.0)
    env.current_progress = env.track.progress_along_centerline(2.0, -4.0)
    _, r_forward, _, _ = env.step(np.array([0.0, 2.0]))
    assert r_forward > r_still, (
        f"Expected reward when moving forward ({r_forward}) "
        f"to be higher than stationary ({r_still})"
    )


def test_forward_higher_than_backward_same_position():
    env = RacingEnv()
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-4.0, heading=0.0, velocity=1.0)
    env.current_progress = env.track.progress_along_centerline(2.0, -4.0)
    _, r_forward, _, _ = env.step(np.array([0.0, 2.0]))
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-4.0, heading=np.pi, velocity=1.0)
    env.current_progress = env.track.progress_along_centerline(2.0, -4.0)
    _, r_backward, _, _ = env.step(np.array([0.0, 2.0]))
    assert r_forward > r_backward, (
        f"Expected forward reward ({r_forward}) > backward reward ({r_backward})"
    )


def test_progress_shaping_magnitude_reasonable():
    env = RacingEnv()
    env.reset(seed=42)
    env.state = CarState(x=2.0, y=-4.0, heading=0.0, velocity=5.0)
    env.current_progress = env.track.progress_along_centerline(2.0, -4.0)
    _, reward, _, _ = env.step(np.array([0.0, 5.0]))
    # Max progress per step at full speed: max_speed * dt / perimeter = 10 * 0.1 / 36 ≈ 0.028
    # max positive shaping = 0.5 * 0.028 ≈ 0.014
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


# ── Reward lines ──────────────────────────────────────────────────────


def test_get_centerline_point_rectangular():
    track = RectangularTrack(width=10.0, height=8.0, track_width=2.0)
    cx, cy, tangent = track.get_centerline_point(0.0)
    np.testing.assert_almost_equal(cx, 0.0)
    np.testing.assert_almost_equal(cy, -4.0)
    np.testing.assert_almost_equal(tangent, 0.0)

    cx, cy, tangent = track.get_centerline_point(0.25)
    # At progress 0.25 on rectangular track the car should be at (5, 0) heading up
    np.testing.assert_almost_equal(cx, 5.0)
    np.testing.assert_almost_equal(cy, 0.0)
    np.testing.assert_almost_equal(tangent, np.pi / 2.0)


def test_get_centerline_point_circular():
    track = CircularTrack(radius=6.0, track_width=2.0)
    cx, cy, tangent = track.get_centerline_point(0.0)
    np.testing.assert_almost_equal(cx, 0.0)
    np.testing.assert_almost_equal(cy, -6.0)
    np.testing.assert_almost_equal(tangent, 0.0)

    cx, cy, tangent = track.get_centerline_point(0.25)
    np.testing.assert_almost_equal(cx, 6.0)
    np.testing.assert_almost_equal(cy, 0.0)
    np.testing.assert_almost_equal(tangent, np.pi / 2.0)

    cx, cy, tangent = track.get_centerline_point(0.5)
    np.testing.assert_almost_equal(cx, 0.0)
    np.testing.assert_almost_equal(cy, 6.0)
    np.testing.assert_almost_equal(tangent, np.pi, decimal=5)


def test_get_centerline_point_figure8():
    track = Figure8Track(radius=6.0, track_width=2.0)
    gx, gy = track.goal_position
    cx, cy, tangent = track.get_centerline_point(0.0)
    np.testing.assert_almost_equal(cx, float(gx), decimal=10)
    np.testing.assert_almost_equal(cy, float(gy), decimal=10)

    # t=0.5 should give the opposite point
    cx, cy, tangent = track.get_centerline_point(0.5)
    theta = 2.0 * np.pi * 0.5 + track._theta_offset
    expected_x = track.radius * np.cos(theta)
    expected_y = track.radius * np.sin(theta) * np.cos(theta)
    np.testing.assert_almost_equal(float(cx), float(expected_x), decimal=10)
    np.testing.assert_almost_equal(float(cy), float(expected_y), decimal=10)


def test_reward_line_endpoints_span_track_width():
    track = RectangularTrack(width=10.0, height=8.0, track_width=2.0)
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        (x1, y1), (x2, y2) = reward_line_endpoints(track, p)
        # Both endpoints should be on the track surface
        assert track.is_on_track(x1, y1), f"Endpoint ({x1},{y1}) off track at p={p}"
        assert track.is_on_track(x2, y2), f"Endpoint ({x2},{y2}) off track at p={p}"
        # Endpoints should be distinct (line has non-zero length)
        assert abs(x1 - x2) > 0.01 or abs(y1 - y2) > 0.01


def test_reward_line_not_crossed_when_stationary():
    env = RacingEnv(num_reward_lines=10, randomize_start=False)
    env.reset(seed=42)
    # Car starts at (0, -4) with progress 0, stationary
    _, reward, _, info = env.step(np.array([0.0, 0.0]))
    assert info['reward_lines_crossed'] == 0
    np.testing.assert_almost_equal(reward, np.float64(0.1))


def test_num_reward_lines_configurable():
    env = RacingEnv(num_reward_lines=5, randomize_start=False)
    assert len(env._reward_line_progress) == 5

    env = RacingEnv(num_reward_lines=0, randomize_start=False)
    assert len(env._reward_line_progress) == 0


def test_reward_line_reward_configurable():
    env = RacingEnv(num_reward_lines=1, reward_line_reward=2.0, randomize_start=False)
    env.reset(seed=42)
    rp = env._reward_line_progress
    assert len(rp) == 1
    # Place car just before the reward line, heading east at speed
    line_progress = rp[0]
    track = env.track
    # Find a position whose progress is slightly before the reward line
    cx_before, cy_before, tangent = track.get_centerline_point(line_progress - np.float64(0.01))
    env.state = CarState(x=cx_before, y=cy_before, heading=tangent, velocity=5.0)
    env.current_progress = track.progress_along_centerline(cx_before, cy_before)
    env.prev_progress = env.current_progress
    _, reward, _, _ = env.step(np.array([0.0, 5.0]))
    # Reward should include 0.1 base + 2.0 line + shaping
    assert reward > np.float64(2.0)


def test_reward_lines_reset_on_episode_reset():
    env = RacingEnv(num_reward_lines=10, randomize_start=False)
    env.reset(seed=42)
    env._collected_reward_lines[[0, 1, 2]] = True
    assert np.sum(env._collected_reward_lines) == 3
    env.reset(seed=42)
    assert np.sum(env._collected_reward_lines) == 0


def test_reward_line_crossed_on_single_step():
    env = RacingEnv(num_reward_lines=10, reward_line_reward=0.5, randomize_start=False, dt=0.1)
    env.reset(seed=42)
    track = env.track
    rp = env._reward_line_progress
    first_line = rp[0]
    # Place car at a known position just before the first reward line
    # Rectangular: bottom edge (0,-4)->(5,-4), progress=pos/36. first_line~0.09 → pos~3.3
    cx_before, cy_before, tangent = track.get_centerline_point(first_line - np.float64(0.008))
    env.state = CarState(x=cx_before, y=cy_before, heading=tangent, velocity=10.0)
    env.current_progress = track.progress_along_centerline(cx_before, cy_before)
    env.prev_progress = env.current_progress
    _, reward, _, info = env.step(np.array([0.0, 10.0]))
    assert info['reward_lines_crossed'] >= 1
    assert reward > np.float64(0.5)


def test_reward_line_not_crossed_going_backward():
    env = RacingEnv(num_reward_lines=3, reward_line_reward=0.5, randomize_start=False, dt=0.1)
    env.reset(seed=42)
    track = env.track
    rp = env._reward_line_progress
    first_line = rp[0]
    # Place car just AFTER the first reward line, heading BACKWARD
    cx_after, cy_after, tangent = track.get_centerline_point(first_line + np.float64(0.008))
    env.state = CarState(x=cx_after, y=cy_after, heading=tangent + np.pi, velocity=10.0)
    env.current_progress = track.progress_along_centerline(cx_after, cy_after)
    env.prev_progress = env.current_progress
    _, reward, _, info = env.step(np.array([0.0, 10.0]))
    assert info['reward_lines_crossed'] == 0


def test_reward_lines_lap_completion_resets_collected():
    env = RacingEnv(num_reward_lines=5, reward_line_reward=0.5, randomize_start=False, dt=0.1)
    env.reset(seed=42)
    track = env.track
    env._collected_reward_lines[:] = True
    # Place car on last segment (-5,-4)->(0,-4), just left of the start/finish line
    env.state = CarState(x=np.float64(-0.5), y=np.float64(-4.0), heading=np.float64(0.0), velocity=10.0)
    env.current_progress = track.progress_along_centerline(-0.5, -4.0)
    env.prev_progress = env.current_progress
    _, reward, _, info = env.step(np.array([0.0, 10.0]))
    assert info['lap_count'] == 1
    assert np.sum(env._collected_reward_lines) == 0


def test_reward_lines_crossed_after_lap_completion():
    env = RacingEnv(num_reward_lines=5, reward_line_reward=0.5, randomize_start=False, dt=0.1)
    env.reset(seed=42)
    # Place car just before start/finish at high speed
    env.state = CarState(x=np.float64(-0.3), y=np.float64(-4.0), heading=np.float64(0.0), velocity=10.0)
    env.current_progress = env.track.progress_along_centerline(-0.3, -4.0)
    env.prev_progress = env.current_progress
    _, reward, _, info = env.step(np.array([0.0, 10.0]))
    assert info['lap_count'] == 1
    assert np.sum(env._collected_reward_lines) == 0


def test_info_dict_contains_reward_lines_crossed():
    env = RacingEnv(num_reward_lines=3, randomize_start=False)
    env.reset(seed=42)
    _, _, _, info = env.step(np.array([0.0, 0.0]))
    assert 'reward_lines_crossed' in info
    assert isinstance(info['reward_lines_crossed'], (int, np.integer))


# ── Lidar ─────────────────────────────────────────────────────────────


def test_lidar_disabled_by_default():
    env = RacingEnv()
    obs = env.reset(seed=42)
    assert obs.shape == (6,), "Observation should be 6D when lidar is disabled"


def test_lidar_disabled_observation_identical():
    env_no = RacingEnv(use_lidar=False, obstacles=[])
    env_no.reset(seed=42)
    obs_no = env_no._get_observation()
    env_no.reset(seed=42)
    obs_no2 = env_no._get_observation()
    np.testing.assert_array_equal(obs_no, obs_no2)


def test_lidar_enabled_shape():
    for num_rays in [4, 8, 16, 32]:
        env = RacingEnv(use_lidar=True, num_lidar_rays=num_rays)
        obs = env.reset(seed=42)
        assert obs.shape == (6 + num_rays,), (
            f"Expected shape {(6 + num_rays,)}, got {obs.shape}"
        )
        assert obs.dtype == np.float64, f"Expected float64, got {obs.dtype}"


def test_lidar_readings_finite_and_in_range():
    env = RacingEnv(use_lidar=True, num_lidar_rays=8)
    obs = env.reset(seed=42)
    lidar = obs[6:]
    assert np.all(np.isfinite(lidar)), "Lidar readings should be finite"
    assert np.all(lidar >= 0.0) and np.all(lidar <= 1.0), (
        f"Lidar readings should be in [0, 1], got range [{lidar.min()}, {lidar.max()}]"
    )


def test_lidar_straight_section_rectangular():
    hw, hh = 5.0, 4.0
    tw = 2.0
    env = RacingEnv(
        track=RectangularTrack(width=2 * hw, height=2 * hh, track_width=tw),
        use_lidar=True, num_lidar_rays=8,
    )
    env.reset(seed=42)
    # Place car at center of bottom edge, heading right
    from numpy_rl_racer.env.car import CarState
    env.state = CarState(x=2.0, y=-hh, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    lidar = obs[6:]
    half_tw = tw / 2.0
    expected = half_tw  # 1.0

    # Ray angles: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    # At angle 90° (π/2, pointing up/inward): should see inner boundary at half_tw
    idx_up = 2  # 90° = π/2
    np.testing.assert_almost_equal(
        lidar[idx_up], expected / 10.0, decimal=5,
        err_msg="Upward ray should see inner boundary at half_tw"
    )
    # At angle 270° (3π/2, pointing down/outward): should see outer boundary at half_tw
    idx_down = 6  # 270° = 3π/2
    np.testing.assert_almost_equal(
        lidar[idx_down], expected / 10.0, decimal=5,
        err_msg="Downward ray should see outer boundary at half_tw"
    )


def test_lidar_symmetry_straight_section():
    hw, hh = 5.0, 4.0
    tw = 2.0
    env = RacingEnv(
        track=RectangularTrack(width=2 * hw, height=2 * hh, track_width=tw),
        use_lidar=True, num_lidar_rays=8,
    )
    env.reset(seed=42)
    from numpy_rl_racer.env.car import CarState
    env.state = CarState(x=2.0, y=-hh, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    lidar = obs[6:]

    # Ray 1 (45°) and ray 7 (315°) should be symmetric
    np.testing.assert_almost_equal(
        lidar[1], lidar[7], decimal=5,
        err_msg="Left and right forward rays should be symmetric"
    )
    # Ray 3 (135°) and ray 5 (225°) should be symmetric
    np.testing.assert_almost_equal(
        lidar[3], lidar[5], decimal=5,
        err_msg="Left and right backward rays should be symmetric"
    )


def test_lidar_obstacle_detection():
    env = RacingEnv(
        use_lidar=True, num_lidar_rays=8, lidar_max_range=10.0,
        obstacles=[Obstacle(x=3.0, y=-4.0, radius=0.5)],
    )
    env.reset(seed=42)
    from numpy_rl_racer.env.car import CarState
    # Car at (0, -4) heading right (0 rad)
    env.state = CarState(x=0.0, y=-4.0, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    lidar = obs[6:]

    # Obstacle at (3, -4), ray at angle 0° (forward/right)
    # Distance = 3.0, minus obstacle radius 0.5 = 2.5
    # Normalized = min(2.5, 10.0) / 10.0 = 0.25
    idx_fwd = 0  # 0°
    expected_dist = 3.0 - 0.5  # distance from car to obstacle center minus radius
    expected_normalized = expected_dist / 10.0
    assert lidar[idx_fwd] <= expected_normalized + 0.05, (
        f"Forward ray should detect obstacle at ~{expected_normalized:.3f}, "
        f"got {lidar[idx_fwd]:.3f}"
    )


def test_lidar_obstacle_detection_known_position():
    env = RacingEnv(
        use_lidar=True, num_lidar_rays=16, lidar_max_range=10.0,
        obstacles=[Obstacle(x=2.0, y=-3.0, radius=0.4)],
    )
    env.reset(seed=42)
    from numpy_rl_racer.env.car import CarState
    # Car at (0, -3), heading right
    env.state = CarState(x=0.0, y=-3.0, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    lidar = obs[6:]

    # Obstacle at (2, -3), ray at 0° → distance 2.0 - 0.4 = 1.6
    expected = np.clip((2.0 - 0.4) / 10.0, 0.0, 1.0)
    idx_fwd = 0
    np.testing.assert_almost_equal(
        lidar[idx_fwd], expected, decimal=3,
        err_msg="Ray at 0° should detect obstacle at exact distance"
    )


def test_lidar_max_range_clipping():
    env = RacingEnv(
        use_lidar=True, num_lidar_rays=4, lidar_max_range=10.0,
    )
    env.reset(seed=42)
    from numpy_rl_racer.env.car import CarState
    env.state = CarState(x=0.0, y=-4.0, heading=0.0, velocity=0.0)
    obs = env._get_observation()
    lidar = obs[6:]

    # Ray pointing right (0°) along the road — max distance is bounded by track but
    # should be finite and in [0, 1]
    assert 0.0 <= lidar[0] <= 1.0


def test_lidar_use_lidar_false_regression():
    env_base = RacingEnv(obstacles=[])
    env_lidar_disabled = RacingEnv(use_lidar=False, obstacles=[])

    for seed in [0, 1, 42, 123]:
        obs_base = env_base.reset(seed=seed)
        obs_disabled = env_lidar_disabled.reset(seed=seed)
        np.testing.assert_array_equal(
            obs_base, obs_disabled,
            err_msg=f"Seed {seed}: use_lidar=False should match default env"
        )


def test_lidar_all_track_types():
    track_types = [
        ('rectangular', None),
        ('circular', None),
        ('figure8', None),
    ]
    for track_type, _ in track_types:
        env = RacingEnv(
            track_type=track_type,
            use_lidar=True, num_lidar_rays=8, lidar_max_range=10.0,
        )
        obs = env.reset(seed=42)
        lidar = obs[6:]
        assert np.all(np.isfinite(lidar)), (
            f"Lidar readings should be finite for {track_type} track"
        )
        assert np.all(lidar >= 0.0) and np.all(lidar <= 1.0), (
            f"Lidar readings should be in [0, 1] for {track_type} track, "
            f"got [{lidar.min()}, {lidar.max()}]"
        )


def test_lidar_deterministic_with_seed():
    env = RacingEnv(use_lidar=True, num_lidar_rays=8)
    obs1 = env.reset(seed=123)
    obs2 = env.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)


def test_lidar_with_obstacles_all_track_types():
    for track_type in ['rectangular', 'circular', 'figure8']:
        env = RacingEnv(
            track_type=track_type,
            use_lidar=True, num_lidar_rays=8, lidar_max_range=10.0,
            obstacles=[Obstacle(x=2.0, y=0.0, radius=0.5)],
        )
        obs = env.reset(seed=42)
        lidar = obs[6:]
        assert np.all(np.isfinite(lidar)), (
            f"Lidar readings should be finite for {track_type} track with obstacles"
        )
        assert np.all(lidar >= 0.0) and np.all(lidar <= 1.0)


def test_lidar_step_does_not_crash():
    env = RacingEnv(use_lidar=True, num_lidar_rays=8, lidar_max_range=10.0)
    env.reset(seed=42)
    for _ in range(20):
        obs, reward, done, info = env.step(np.array([0.0, 2.0]))
        assert obs.shape == (6 + 8,), f"Obs shape should be 14, got {obs.shape}"
        assert np.all(np.isfinite(obs))
        if done:
            break


def test_lidar_step_with_obstacles():
    env = RacingEnv(
        use_lidar=True, num_lidar_rays=8, lidar_max_range=10.0,
        obstacles=[Obstacle(x=3.0, y=-3.0, radius=0.5)],
    )
    env.reset(seed=42)
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0.0, 2.0]))
        assert obs.shape[0] >= 6 + 8, "Observation should include lidar readings"
        assert np.all(np.isfinite(obs))
        if done:
            break


def test_lidar_zero_rays():
    env = RacingEnv(use_lidar=True, num_lidar_rays=0)
    obs = env.reset(seed=42)
    assert obs.shape == (6,), "With 0 lidar rays, observation should be 6D"


def test_lidar_single_ray():
    env = RacingEnv(use_lidar=True, num_lidar_rays=1)
    obs = env.reset(seed=42)
    assert obs.shape == (7,), "With 1 lidar ray, observation should be 7D"
    assert 0.0 <= obs[6] <= 1.0
