import numpy as np
import pytest

from numpy_rl_racer.env.racing_env import Obstacle, RacingEnv
from numpy_rl_racer.env.wrappers import ActionRepeatEnv, EpisodeMonitor, TrackPoolEnv
from numpy_rl_racer.env.car import CarState


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


def test_compatible_with_procedural_track():
    env = RacingEnv()
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


class _MockTrack:
    """Mock track exposing the fields EpisodeMonitor reads (track_width, centerline_info)."""

    def __init__(self, track_width, env):
        self.track_width = track_width
        self._env = env

    def centerline_info(self, x, y):
        # Reverse the state-mode observation mapping so the mock's obs[4]
        # (dist_to_edge_normalized) reproduces the same dist_to_edge values the
        # old obs-indexing implementation derived.
        half_tw = float(self.track_width) / 2.0
        obs = self._env._current_obs
        dist_to_edge_normalized = float(obs[4])
        dist_to_centerline = half_tw - dist_to_edge_normalized * half_tw
        return (np.float64(dist_to_centerline), np.float64(0.0))


class _MockEnv:
    """Helper: deterministic mock env for precise stat verification."""

    def __init__(self, obs_list, rewards, dones, infos, obstacles=None, track_width=2.0):
        self._obs_list = obs_list
        self._rewards = rewards
        self._dones = dones
        self._infos = infos
        self._idx = 0
        self.obstacles = obstacles if obstacles is not None else []
        self.state = None
        self._current_obs = None
        self.track = _MockTrack(track_width, self)

    def step(self, action):
        i = self._idx
        self._idx += 1
        obs = self._obs_list[i]
        self._apply_obs(obs)
        return obs, self._rewards[i], self._dones[i], self._infos[i]

    def reset(self, seed=None):
        self._idx = 0
        obs = self._obs_list[0]
        self._apply_obs(obs)
        return obs

    def _apply_obs(self, obs):
        self._current_obs = obs
        self.state = CarState(
            x=float(obs[0]),
            y=float(obs[1]),
            heading=0.0,
            velocity=float(obs[3]),
        )


class TestEpisodeMonitor:
    def test_length_and_total_reward(self):
        obs = np.zeros(6, dtype=np.float64)
        env = _MockEnv([obs] * 3, [1.5] * 3, [False, False, True], [{"lap_count": 0}] * 3)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        for a in [[0.0, 0.0]] * 3:
            monitor.step(np.array(a))
        stats = monitor.get_episode_stats()
        assert stats["length"] == 3
        assert stats["total_reward"] == 4.5

    def test_avg_and_max_speed(self):
        obs_list = []
        for v in [0.5, 2.0, 1.5, 3.0]:
            o = np.zeros(6, dtype=np.float64)
            o[3] = v
            obs_list.append(o)
        env = _MockEnv(obs_list, [0.0] * 4, [False] * 4, [{"lap_count": 0}] * 4)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        for a in [[0.0, 0.0]] * 4:
            monitor.step(np.array(a))
        stats = monitor.get_episode_stats()
        assert stats["avg_speed"] == 1.75
        assert stats["max_speed"] == 3.0

    def test_min_dist_to_edge(self):
        obs_list = []
        for d in [1.0, 0.5, 0.8, 0.3]:
            o = np.zeros(6, dtype=np.float64)
            o[4] = d
            obs_list.append(o)
        env = _MockEnv(obs_list, [0.0] * 4, [False] * 4, [{"lap_count": 0}] * 4, track_width=2.0)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        for a in [[0.0, 0.0]] * 4:
            monitor.step(np.array(a))
        stats = monitor.get_episode_stats()
        assert stats["min_dist_to_edge"] == pytest.approx(0.3)

    def test_off_track_steps_counted(self):
        obs_list = []
        for d in [0.02, 1.0, 0.0, 0.5]:
            o = np.zeros(6, dtype=np.float64)
            o[4] = d
            obs_list.append(o)
        env = _MockEnv(obs_list, [0.0] * 4, [False] * 4, [{"lap_count": 0}] * 4, track_width=2.0)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        for a in [[0.0, 0.0]] * 4:
            monitor.step(np.array(a))
        stats = monitor.get_episode_stats()
        assert stats["off_track_steps"] == 2

    def test_obstacle_collisions_detected(self):
        obstacle = Obstacle(2.0, 2.0, 0.5)
        obs_hit = np.zeros(6, dtype=np.float64)
        obs_hit[0], obs_hit[1] = 2.0, 2.0
        obs_miss = np.zeros(6, dtype=np.float64)
        obs_miss[0], obs_miss[1] = 10.0, 10.0
        obs_near = np.zeros(6, dtype=np.float64)
        obs_near[0], obs_near[1] = 2.5, 2.0  # dist = 0.5, threshold = 0.8 → collision
        obs_list = [obs_hit, obs_miss, obs_near]
        env = _MockEnv(
            obs_list, [0.0] * 3, [False] * 3, [{"lap_count": 0}] * 3,
            obstacles=[obstacle], track_width=2.0,
        )
        monitor = EpisodeMonitor(env)
        monitor.reset()
        for a in [[0.0, 0.0]] * 3:
            monitor.step(np.array(a))
        stats = monitor.get_episode_stats()
        assert stats["obstacle_collisions"] == 2

    def test_laps_completed(self):
        infos = [{"lap_count": 0}, {"lap_count": 0}, {"lap_count": 0}, {"lap_count": 1}, {"lap_count": 1}]
        obs = np.zeros(6, dtype=np.float64)
        env = _MockEnv([obs] * 5, [0.0] * 5, [False] * 5, infos)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        for a in [[0.0, 0.0]] * 5:
            monitor.step(np.array(a))
        stats = monitor.get_episode_stats()
        assert stats["laps_completed"] == 1

    def test_distance_traveled(self):
        positions = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (6.0, 4.0)]
        obs_list = []
        for x, y in positions:
            o = np.zeros(6, dtype=np.float64)
            o[0], o[1] = x, y
            obs_list.append(o)
        env = _MockEnv(obs_list, [0.0] * 4, [False] * 4, [{"lap_count": 0}] * 4)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        for a in [[0.0, 0.0]] * 4:
            monitor.step(np.array(a))
        stats = monitor.get_episode_stats()
        # reset position (0,0), step1→(3,0):3, step2→(3,4):4, step3→(6,4):3
        assert stats["distance_traveled"] == pytest.approx(10.0)

    def test_accumulation_monotonic(self):
        obs = np.zeros(6, dtype=np.float64)
        obs[3] = 1.0
        env = _MockEnv([obs] * 5, [1.0] * 5, [False] * 5, [{"lap_count": 0}] * 5)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        prev_len = 0
        prev_reward = 0.0
        for i in range(5):
            monitor.step(np.array([0.0, 0.0]))
            stats = monitor.get_episode_stats()
            assert stats["length"] == i + 1 > prev_len
            assert stats["total_reward"] == i + 1 > prev_reward
            assert stats["distance_traveled"] >= 0.0
            prev_len = stats["length"]
            prev_reward = stats["total_reward"]

    def test_episode_boundary(self):
        obs = np.zeros(6, dtype=np.float64)
        obs[3] = 2.0
        env = _MockEnv(
            [obs] * 6,
            [1.0] * 6,
            [False, False, True, False, False, True],
            [{"lap_count": 0}] * 6,
        )
        monitor = EpisodeMonitor(env)
        monitor.reset()
        monitor.step(np.array([0.0, 0.0]))
        monitor.step(np.array([0.0, 0.0]))
        monitor.step(np.array([0.0, 0.0]))  # done=True
        stats1 = monitor.get_episode_stats()
        assert stats1["length"] == 3
        assert stats1["total_reward"] == 3.0

        # Second episode follows done=True without explicit reset
        monitor.step(np.array([0.0, 0.0]))
        monitor.step(np.array([0.0, 0.0]))  # done=True
        stats2 = monitor.get_episode_stats()
        assert stats2["length"] == 2
        assert stats2["total_reward"] == 2.0

    def test_reset_clears_stats(self):
        obs = np.zeros(6, dtype=np.float64)
        env = _MockEnv([obs] * 4, [1.0] * 4, [False] * 4, [{"lap_count": 0}] * 4)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        monitor.step(np.array([0.0, 0.0]))
        monitor.step(np.array([0.0, 0.0]))
        stats_before = monitor.get_episode_stats()
        assert stats_before["length"] == 2
        monitor.reset()
        stats_after = monitor.get_episode_stats()
        assert stats_after["length"] == 0
        assert stats_after["total_reward"] == 0.0
        assert stats_after["off_track_steps"] == 0
        assert stats_after["obstacle_collisions"] == 0
        assert stats_after["laps_completed"] == 0

    def test_info_dict_contains_episode_monitor_keys(self):
        env = RacingEnv()
        monitor = EpisodeMonitor(env)
        monitor.reset(seed=42)
        _, _, _, info = monitor.step(np.array([0.0, 0.0]))
        assert "episode_monitor/length" in info
        assert "episode_monitor/total_reward" in info
        assert "episode_monitor/avg_speed" in info
        assert "episode_monitor/max_speed" in info
        assert "episode_monitor/min_dist_to_edge" in info
        assert "episode_monitor/off_track_steps" in info
        assert "episode_monitor/obstacle_collisions" in info
        assert "episode_monitor/laps_completed" in info
        assert "episode_monitor/distance_traveled" in info

    def test_get_episode_stats_mid_episode(self):
        obs_list = []
        for v in [1.0, 2.0]:
            o = np.zeros(6, dtype=np.float64)
            o[3] = v
            obs_list.append(o)
        env = _MockEnv(obs_list, [0.5, 1.5], [False, False], [{"lap_count": 0}] * 2)
        monitor = EpisodeMonitor(env)
        monitor.reset()
        monitor.step(np.array([0.0, 0.0]))
        mid_stats = monitor.get_episode_stats()
        assert mid_stats["length"] == 1
        assert mid_stats["total_reward"] == 0.5
        assert mid_stats["avg_speed"] == 1.0
        assert mid_stats["max_speed"] == 1.0

        monitor.step(np.array([0.0, 0.0]))
        final_stats = monitor.get_episode_stats()
        assert final_stats["length"] == 2
        assert final_stats["total_reward"] == 2.0
        assert final_stats["avg_speed"] == 1.5
        assert final_stats["max_speed"] == 2.0

    def test_compatible_with_procedural_track(self):
        env = RacingEnv()
        monitor = EpisodeMonitor(env)
        monitor.reset(seed=42)
        for _ in range(5):
            obs, reward, done, info = monitor.step(np.array([0.0, 0.5]))
            assert "episode_monitor/length" in info
            if done:
                monitor.reset(seed=42)
        stats = monitor.get_episode_stats()
        assert stats["length"] >= 0

    def test_compatible_with_obstacles(self):
        obstacles_env = [Obstacle(0.0, 0.0, 0.5)]
        env = RacingEnv(obstacles=obstacles_env)
        monitor = EpisodeMonitor(env)
        monitor.reset(seed=42)
        for _ in range(5):
            obs, reward, done, info = monitor.step(np.array([0.0, 0.5]))
            assert "episode_monitor/obstacle_collisions" in info
            if done:
                monitor.reset(seed=42)

    def test_getattr_passthrough(self):
        env = RacingEnv()
        monitor = EpisodeMonitor(env)
        assert monitor.track is env.track
        assert monitor.dt == env.dt
        assert monitor.env is env

    def test_local_mode_avg_and_max_speed_match_state(self):
        env = RacingEnv(observation_mode="local")
        monitor = EpisodeMonitor(env)
        monitor.reset(seed=42)
        velocities = []
        for _ in range(10):
            monitor.step(np.array([0.0, 1.0]))
            velocities.append(float(env.state.velocity))
        stats = monitor.get_episode_stats()
        assert stats["avg_speed"] == pytest.approx(np.mean(velocities))
        assert stats["max_speed"] == pytest.approx(np.max(velocities))

    def test_local_mode_distance_traveled_matches_state_displacements(self):
        env = RacingEnv(observation_mode="local")
        monitor = EpisodeMonitor(env)
        monitor.reset(seed=42)
        xs = [float(env.state.x)]
        ys = [float(env.state.y)]
        for _ in range(10):
            monitor.step(np.array([0.0, 1.0]))
            xs.append(float(env.state.x))
            ys.append(float(env.state.y))
        expected = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))
        stats = monitor.get_episode_stats()
        assert stats["distance_traveled"] == pytest.approx(expected)

    def test_local_mode_min_dist_to_edge_matches_centerline(self):
        env = RacingEnv(observation_mode="local", randomize_start=False)
        monitor = EpisodeMonitor(env)
        monitor.reset(seed=42)
        expected_min = np.inf
        for _ in range(10):
            monitor.step(np.array([0.0, 1.0]))
            dist_to_centerline, _ = env.track.centerline_info(
                float(env.state.x), float(env.state.y)
            )
            half_tw = float(env.track.track_width) / 2.0
            dist_to_edge = half_tw - float(dist_to_centerline)
            expected_min = min(expected_min, dist_to_edge)
        stats = monitor.get_episode_stats()
        assert stats["min_dist_to_edge"] == pytest.approx(expected_min)

    def test_local_mode_min_dist_to_edge_decreases_toward_edge(self):
        env = RacingEnv(observation_mode="local", randomize_start=False)
        monitor = EpisodeMonitor(env)
        monitor.reset(seed=42)
        # Steer hard toward one side; the car drifts laterally toward an edge,
        # so the reported min_dist_to_edge should eventually drop below the
        # starting value as the episode progresses (it is a running min).
        prev_min = monitor.get_episode_stats()["min_dist_to_edge"]
        decreased_at_some_point = False
        for _ in range(50):
            monitor.step(np.array([0.5, 1.0]))
            current_min = monitor.get_episode_stats()["min_dist_to_edge"]
            if current_min < prev_min:
                decreased_at_some_point = True
            prev_min = current_min
            if env.state is None:
                break
        assert decreased_at_some_point

    def test_track_pool_episode_monitor_local_mode_across_resets(self):
        pool = TrackPoolEnv(track_seeds=[0, 1, 2], observation_mode="local")
        monitor = EpisodeMonitor(pool)
        for _ in range(2):
            monitor.reset(seed=42)
            for _ in range(5):
                obs, reward, done, info = monitor.step(np.array([0.0, 1.0]))
                assert "episode_monitor/length" in info
                if done:
                    break
        stats = monitor.get_episode_stats()
        assert stats["length"] >= 0
        assert stats["avg_speed"] >= 0.0
        assert stats["distance_traveled"] >= 0.0


class TestTrackPoolEnv:
    def test_track_pool_constructs_multiple_envs(self):
        wrapper = TrackPoolEnv(track_seeds=[0, 1, 2])
        assert len(wrapper.envs) == 3
        for env in wrapper.envs:
            assert isinstance(env, RacingEnv)

    def test_track_pool_round_robin_cycles(self):
        wrapper = TrackPoolEnv(track_seeds=[0, 1, 2], mode="round_robin")
        for _ in range(2):
            for expected in wrapper.envs:
                wrapper.reset(seed=42)
                assert wrapper.env is expected

    def test_track_pool_random_reproducible(self):
        w1 = TrackPoolEnv(track_seeds=[0, 1, 2, 3], mode="random", seed=7)
        w2 = TrackPoolEnv(track_seeds=[0, 1, 2, 3], mode="random", seed=7)
        seq1, seq2 = [], []
        for _ in range(10):
            w1.reset(seed=42)
            w2.reset(seed=42)
            seq1.append(w1.envs.index(w1.env))
            seq2.append(w2.envs.index(w2.env))
        assert seq1 == seq2

    def test_track_pool_reset_returns_observation(self):
        wrapper = TrackPoolEnv(track_seeds=[0, 1, 2])
        obs = wrapper.reset(seed=42)
        assert obs.shape == (wrapper.observation_dim,)
        assert obs.dtype == np.float64

    def test_track_pool_step_delegates(self):
        wrapper = TrackPoolEnv(track_seeds=[0, 1, 2])
        wrapper.reset(seed=42)
        result = wrapper.step(np.array([0.0, 1.0]))
        assert isinstance(result, tuple) and len(result) == 4
        obs, reward, done, info = result
        assert obs.shape == (wrapper.observation_dim,)
        assert obs.dtype == np.float64
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(reward, (float, np.floating))
        assert isinstance(info, dict)

    def test_track_pool_getattr_passthrough(self):
        wrapper = TrackPoolEnv(track_seeds=[0, 1, 2])
        wrapper.reset(seed=42)
        assert wrapper.track is wrapper.env.track
        assert wrapper.observation_dim == wrapper.env.observation_dim
        assert wrapper.obstacles is wrapper.env.obstacles

    def test_track_pool_empty_seeds_raises(self):
        with pytest.raises(ValueError, match="track_seeds"):
            TrackPoolEnv(track_seeds=[])

    def test_track_pool_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            TrackPoolEnv(track_seeds=[0, 1], mode="bogus")

    def test_track_pool_obstacles_forwarded(self):
        obstacles = [Obstacle(0.0, 0.0, 0.5)]
        wrapper = TrackPoolEnv(track_seeds=[0, 1], obstacles=obstacles)
        for env in wrapper.envs:
            assert env.obstacles == obstacles
        assert wrapper.observation_dim == 8

    def test_track_pool_compatible_with_episode_monitor(self):
        wrapper = TrackPoolEnv(track_seeds=[0, 1])
        monitor = EpisodeMonitor(wrapper)
        monitor.reset(seed=42)
        info = {}
        for _ in range(5):
            obs, reward, done, info = monitor.step(np.array([0.0, 0.5]))
            assert "episode_monitor/length" in info
            if done:
                monitor.reset(seed=42)
