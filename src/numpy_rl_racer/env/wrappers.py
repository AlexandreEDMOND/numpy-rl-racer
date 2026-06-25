import numpy as np

from .racing_env import CAR_COLLISION_RADIUS


class EpisodeMonitor:
    def __init__(self, env):
        self.env = env
        self._reset_stats()
        self._episode_done = False

    def _reset_stats(self):
        self._length = 0
        self._total_reward = 0.0
        self._speeds = []
        self._max_speed = 0.0
        self._min_dist_to_edge = np.inf
        self._off_track_steps = 0
        self._obstacle_collisions = 0
        self._prev_lap_count = 0
        self._laps_completed = 0
        self._prev_position = None
        self._distance_traveled = 0.0

    def _check_obstacle_collision(self, x, y):
        for obs in self.env.obstacles:
            dx = x - obs.x
            dy = y - obs.y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < CAR_COLLISION_RADIUS + obs.radius:
                return True
        return False

    def _compute_stats(self):
        avg_speed = float(np.mean(self._speeds)) if self._speeds else 0.0
        min_dist = float(self._min_dist_to_edge)
        if not np.isfinite(min_dist):
            min_dist = 0.0
        max_speed = self._max_speed if np.isfinite(self._max_speed) else 0.0
        return {
            'length': self._length,
            'total_reward': float(self._total_reward),
            'avg_speed': avg_speed,
            'max_speed': float(max_speed),
            'min_dist_to_edge': min_dist,
            'off_track_steps': self._off_track_steps,
            'obstacle_collisions': self._obstacle_collisions,
            'laps_completed': self._laps_completed,
            'distance_traveled': float(self._distance_traveled),
        }

    def get_episode_stats(self):
        return self._compute_stats()

    def reset(self, seed=None):
        self._reset_stats()
        obs = self.env.reset(seed=seed)
        self._prev_position = (float(obs[0]), float(obs[1]))
        self._episode_done = False
        return obs

    def step(self, action):
        if self._episode_done:
            self._reset_stats()
            self._prev_position = None

        obs, reward, done, info = self.env.step(action)

        self._length += 1
        self._total_reward += float(reward)

        velocity = float(obs[3])
        self._speeds.append(velocity)
        if velocity > self._max_speed:
            self._max_speed = velocity

        half_tw = float(self.env.track.track_width) / 2.0
        dist_to_edge = float(obs[4]) * half_tw
        if dist_to_edge < self._min_dist_to_edge:
            self._min_dist_to_edge = dist_to_edge

        if dist_to_edge <= 0.05:
            self._off_track_steps += 1

        if self.env.obstacles:
            x, y = float(obs[0]), float(obs[1])
            if self._check_obstacle_collision(x, y):
                self._obstacle_collisions += 1

        current_laps = info.get('lap_count', 0)
        lap_diff = current_laps - self._prev_lap_count
        if lap_diff > 0:
            self._laps_completed += lap_diff
        self._prev_lap_count = current_laps

        x, y = float(obs[0]), float(obs[1])
        if self._prev_position is not None:
            dx = x - self._prev_position[0]
            dy = y - self._prev_position[1]
            self._distance_traveled += float(np.sqrt(dx * dx + dy * dy))
        self._prev_position = (x, y)

        stats = self._compute_stats()
        for key, value in stats.items():
            info[f'episode_monitor/{key}'] = value

        self._episode_done = done
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class ActionRepeatEnv:
    def __init__(self, env, skip_frames=4):
        if skip_frames < 1:
            raise ValueError(f"skip_frames must be >= 1, got {skip_frames}")
        self.env = env
        self.skip_frames = skip_frames

    def step(self, action):
        total_reward = np.float64(0.0)
        for _ in range(self.skip_frames):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def __getattr__(self, name):
        return getattr(self.env, name)
