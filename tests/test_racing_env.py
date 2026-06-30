import numpy as np
import pytest

from numpy_rl_racer.env.car import CarState
from numpy_rl_racer.env.racing_env import Obstacle, ProceduralTrack, RacingEnv, reward_line_endpoints


def test_procedural_track_seed_determinism():
    t1 = ProceduralTrack(seed=7)
    t2 = ProceduralTrack(seed=7)
    t3 = ProceduralTrack(seed=8)
    np.testing.assert_allclose(t1.centerline_points, t2.centerline_points)
    assert not np.allclose(t1.centerline_points, t3.centerline_points)


def test_procedural_track_validation():
    with pytest.raises(ValueError, match="num_control_points"):
        ProceduralTrack(num_control_points=4)
    with pytest.raises(ValueError, match="radius"):
        ProceduralTrack(radius=0.0)
    with pytest.raises(ValueError, match="track_width"):
        ProceduralTrack(track_width=0.0)
    with pytest.raises(ValueError, match="radial_noise"):
        ProceduralTrack(radial_noise=-0.1)
    with pytest.raises(ValueError, match="smoothing_steps"):
        ProceduralTrack(smoothing_steps=-1)


def test_procedural_track_exposes_boundaries_and_segments():
    track = ProceduralTrack(seed=0)
    assert track.centerline_points.shape[1] == 2
    assert track.outer_boundary.shape[1] == 2
    assert track.inner_boundary.shape[1] == 2
    assert track.boundary_segments.shape[1:] == (2, 2)
    assert track._perimeter > 0.0
    assert track.x_min < track.x_max
    assert track.y_min < track.y_max


def test_centerline_points_are_on_track():
    track = ProceduralTrack(seed=3)
    for p in np.linspace(0.0, 0.95, 20):
        x, y, _ = track.get_centerline_point(p)
        assert track.is_on_track(x, y)


def test_progress_bounds_and_known_start():
    track = ProceduralTrack(seed=0)
    sx, sy, _ = track.start_position
    np.testing.assert_almost_equal(track.progress_along_centerline(sx, sy), 0.0)
    for p in np.linspace(0.0, 0.95, 20):
        x, y, _ = track.get_centerline_point(p)
        progress = track.progress_along_centerline(x, y)
        assert 0.0 <= progress <= 1.0
        assert progress == pytest.approx(p, abs=0.04)


def test_sample_centerline_point_is_reproducible_with_rng():
    track = ProceduralTrack(seed=0)
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    np.testing.assert_allclose(track.sample_centerline_point(rng1), track.sample_centerline_point(rng2))


def test_centerline_info_on_centerline():
    track = ProceduralTrack(seed=0)
    x, y, tangent = track.get_centerline_point(0.25)
    dist, info_tangent = track.centerline_info(x, y)
    assert dist == pytest.approx(0.0, abs=1e-10)
    assert np.isfinite(info_tangent)
    assert abs(np.sin(info_tangent - tangent)) < 0.25


def test_reward_line_endpoints_span_track_width():
    track = ProceduralTrack(seed=0, track_width=2.0)
    (x1, y1), (x2, y2) = reward_line_endpoints(track, 0.25)
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    assert length == pytest.approx(2.0)


def test_reset_returns_numpy_observation():
    env = RacingEnv()
    obs = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert obs.dtype == np.float64


def test_local_observation_dim():
    env = RacingEnv(observation_mode="local")
    obs = env.reset(seed=42)
    assert obs.shape == (9,)
    assert env.observation_dim == 9


def test_step_returns_observation_reward_done_info():
    env = RacingEnv()
    env.reset(seed=42)
    obs, reward, done, info = env.step(np.array([0.0, 2.0]))
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_reset_without_random_start_uses_track_start():
    track = ProceduralTrack(seed=4)
    env = RacingEnv(track=track, randomize_start=False)
    env.reset(seed=42)
    sx, sy, heading = track.start_position
    assert env.state.x == pytest.approx(sx)
    assert env.state.y == pytest.approx(sy)
    assert env.state.heading == pytest.approx(heading)
    assert env.current_progress == np.float64(0.0)


def test_randomized_start_is_on_track():
    track = ProceduralTrack(seed=2)
    env = RacingEnv(track=track, randomize_start=True)
    for seed in range(20):
        env.reset(seed=seed)
        assert track.is_on_track(env.state.x, env.state.y)


def test_respects_seed_determinism():
    env1 = RacingEnv(track=ProceduralTrack(seed=0))
    env2 = RacingEnv(track=ProceduralTrack(seed=0))
    obs1 = env1.reset(seed=123)
    obs2 = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)


def test_progress_reward_penalizes_off_track():
    env = RacingEnv(
        reward_mode="progress",
        observation_mode="local",
        off_track_penalty=5.0,
        randomize_start=False,
    )
    env.reset(seed=0)
    env.state = CarState(x=100.0, y=100.0, heading=0.0, velocity=1.0)
    _, reward, done, _ = env.step(np.array([0.0, 0.0]))
    assert done
    assert reward < 0.0


def test_lidar_readings_are_finite_and_normalized():
    env = RacingEnv(use_lidar=True, num_lidar_rays=8)
    obs = env.reset(seed=42)
    readings = obs[-8:]
    assert np.all(np.isfinite(readings))
    assert np.all((0.0 <= readings) & (readings <= 1.0))


def test_local_ray_readings_are_finite_and_normalized():
    env = RacingEnv(observation_mode="local")
    obs = env.reset(seed=42)
    readings = obs[-5:]
    assert np.all(np.isfinite(readings))
    assert np.all((0.0 <= readings) & (readings <= 1.0))


def test_obstacle_collision_ends_episode():
    track = ProceduralTrack(seed=0)
    sx, sy, _ = track.start_position
    env = RacingEnv(track=track, obstacles=[Obstacle(float(sx), float(sy), 0.5)])
    env.reset(seed=42, randomize_start=False)
    _, _, done, _ = env.step(np.array([0.0, 0.0]))
    assert done


def test_obstacle_observation_shape():
    env = RacingEnv(obstacles=[Obstacle(0.0, 0.0, 0.5)])
    obs = env.reset(seed=42)
    assert obs.shape == (8,)


def test_observation_mode_validation():
    with pytest.raises(ValueError, match="observation_mode"):
        RacingEnv(observation_mode="bad")


def test_reward_mode_validation():
    with pytest.raises(ValueError, match="reward_mode"):
        RacingEnv(reward_mode="bad")
