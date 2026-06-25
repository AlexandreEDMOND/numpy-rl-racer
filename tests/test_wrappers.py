import numpy as np
import pytest

from numpy_rl_racer.env.racing_env import Obstacle, RacingEnv
from numpy_rl_racer.env.wrappers import ActionRepeatEnv, EpisodeMonitor


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


class _MockEnv:
    """Helper: deterministic mock env for precise stat verification."""

    def __init__(self, obs_list, rewards, dones, infos, obstacles=None, track_width=2.0):
        self.track = type("Track", (), {"track_width": track_width})()
        self.obstacles = obstacles if obstacles is not None else []
        self._obs_list = obs_list
        self._rewards = rewards
        self._dones = dones
        self._infos = infos
        self._idx = 0

    def step(self, action):
        i = self._idx
        self._idx += 1
        return self._obs_list[i], self._rewards[i], self._dones[i], self._infos[i]

    def reset(self, seed=None):
        self._idx = 0
        return self._obs_list[0]


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

    @pytest.mark.parametrize("track_type", ["rectangular", "circular", "figure8"])
    def test_compatible_with_all_track_types(self, track_type):
        env = RacingEnv(track_type=track_type)
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
