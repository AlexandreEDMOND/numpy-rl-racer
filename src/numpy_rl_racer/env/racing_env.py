import numpy as np

from .car import CarState, KinematicCar


class RectangularTrack:
    def __init__(self, width=10.0, height=8.0, track_width=2.0):
        self.half_w = np.float64(width / 2.0)
        self.half_h = np.float64(height / 2.0)
        self.track_width = np.float64(track_width)
        self._perimeter = 4.0 * (self.half_w + self.half_h)

    @property
    def goal_position(self):
        return (np.float64(0.0), -self.half_h)

    def progress_along_centerline(self, x, y):
        px, py = np.float64(x), np.float64(y)
        hw, hh = self.half_w, self.half_h

        best_dist = np.inf
        best_cumulative = np.float64(0.0)
        cum_len = np.float64(0.0)

        for x1, y1, x2, y2 in _centerline_edges(hw, hh):
            sx = x2 - x1
            sy = y2 - y1
            seg_len = np.sqrt(sx * sx + sy * sy)
            seg_len_sq = seg_len * seg_len
            if seg_len_sq == 0.0:
                cum_len += seg_len
                continue
            t = ((px - x1) * sx + (py - y1) * sy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            cx = x1 + t * sx
            cy = y1 + t * sy
            dx = px - cx
            dy = py - cy
            dist = np.sqrt(dx * dx + dy * dy)
            cumulative = cum_len + t * seg_len
            if dist < best_dist:
                best_dist = dist
                best_cumulative = cumulative
            cum_len += seg_len

        return best_cumulative / self._perimeter

    def is_on_track(self, x, y):
        px, py = np.float64(x), np.float64(y)
        hw, hh = self.half_w, self.half_h
        tw2 = self.track_width / 2.0

        for x1, y1, x2, y2 in _rectangle_edges(hw, hh):
            if _point_to_segment_dist(px, py, x1, y1, x2, y2) <= tw2:
                return True
        return False


class RacingEnv:
    def __init__(self, track_width=10.0, track_height=8.0, track_road_width=2.0, dt=0.1):
        self.track = RectangularTrack(track_width, track_height, track_road_width)
        self.car = KinematicCar()
        self.dt = np.float64(dt)
        self.state = None
        self.current_progress = np.float64(0.0)
        self.prev_progress = np.float64(0.0)
        self.lap_count = 0

    @property
    def goal_position(self):
        return self.track.goal_position

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = CarState(x=0.0, y=-self.track.half_h, heading=0.0, velocity=0.0)
        self.current_progress = np.float64(0.0)
        self.prev_progress = np.float64(0.0)
        self.lap_count = 0
        return self._get_observation()

    def step(self, action):
        steering, acceleration = np.float64(action[0]), np.float64(action[1])
        self.state = self.car.step(self.state, steering, acceleration, self.dt)
        on_track = self.track.is_on_track(self.state.x, self.state.y)
        done = not on_track

        self.prev_progress = self.current_progress
        self.current_progress = self.track.progress_along_centerline(self.state.x, self.state.y)

        reward = np.float64(0.1 if on_track else -1.0)

        if self.current_progress < self.prev_progress - np.float64(0.5):
            self.lap_count += 1
            reward += np.float64(1.0)

        info = {
            'progress': self.current_progress,
            'lap_count': self.lap_count,
            'goal_position': self.goal_position,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.array(
            [self.state.x, self.state.y, self.state.heading, self.state.velocity],
            dtype=np.float64,
        )


def _centerline_edges(hw, hh):
    return [
        (0, -hh, hw, -hh),
        (hw, -hh, hw, hh),
        (hw, hh, -hw, hh),
        (-hw, hh, -hw, -hh),
        (-hw, -hh, 0, -hh),
    ]


def _rectangle_edges(hw, hh):
    return [
        (-hw, -hh, hw, -hh),
        (hw, -hh, hw, hh),
        (hw, hh, -hw, hh),
        (-hw, hh, -hw, -hh),
    ]


def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    sx = x2 - x1
    sy = y2 - y1
    seg_len_sq = sx * sx + sy * sy
    if seg_len_sq == 0.0:
        dx = px - x1
        dy = py - y1
        return np.sqrt(dx * dx + dy * dy)
    t = ((px - x1) * sx + (py - y1) * sy) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    cx = x1 + t * sx
    cy = y1 + t * sy
    dx = px - cx
    dy = py - cy
    return np.sqrt(dx * dx + dy * dy)
