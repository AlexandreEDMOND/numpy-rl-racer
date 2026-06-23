import numpy as np

from .car import CarState, KinematicCar


PROGRESS_SCALE = np.float64(10.0)
GOAL_REWARD = np.float64(10.0)


class RectangularTrack:
    def __init__(self, width=10.0, height=8.0, track_width=2.0):
        self.half_w = np.float64(width / 2.0)
        self.half_h = np.float64(height / 2.0)
        self.track_width = np.float64(track_width)

    def is_on_track(self, x, y):
        px, py = np.float64(x), np.float64(y)
        hw, hh = self.half_w, self.half_h
        tw2 = self.track_width / 2.0

        for x1, y1, x2, y2 in _rectangle_edges(hw, hh):
            if _point_to_segment_dist(px, py, x1, y1, x2, y2) <= tw2:
                return True
        return False


class RacingEnv:
    def __init__(self, track_width=10.0, track_height=8.0, track_road_width=2.0, dt=0.1,
                 progress_scale=PROGRESS_SCALE, goal_reward=GOAL_REWARD):
        self.track = RectangularTrack(track_width, track_height, track_road_width)
        self.car = KinematicCar()
        self.dt = np.float64(dt)
        self._progress_scale = np.float64(progress_scale)
        self._goal_reward = np.float64(goal_reward)
        self.state = None
        self._prev_progress = None
        self._prev_seg_idx = None
        self._cumulative_progress = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = CarState(x=0.0, y=-self.track.half_h, heading=0.0, velocity=0.0)
        sp, si = _progress_along_centerline(
            0.0, -self.track.half_h, self.track.half_w, self.track.half_h,
        )
        self._prev_progress = sp
        self._prev_seg_idx = si
        self._cumulative_progress = sp
        return self._get_observation()

    def step(self, action):
        steering, acceleration = np.float64(action[0]), np.float64(action[1])
        self.state = self.car.step(self.state, steering, acceleration, self.dt)
        on_track = self.track.is_on_track(self.state.x, self.state.y)
        done = not on_track
        reward = np.float64(0.1 if on_track else -1.0)

        info = {}
        if on_track:
            current_progress, current_seg = _progress_along_centerline(
                self.state.x, self.state.y,
                self.track.half_w, self.track.half_h,
                self._prev_seg_idx,
            )
            raw_delta = current_progress - self._prev_progress

            if raw_delta > 0.5:
                progress_delta = np.float64(0.0)
            elif raw_delta < -0.5:
                progress_delta = raw_delta + np.float64(1.0)
            elif raw_delta >= 0:
                progress_delta = raw_delta
            else:
                progress_delta = np.float64(0.0)

            prev_lap = int(np.floor(self._cumulative_progress))
            self._cumulative_progress += progress_delta
            self._prev_progress = current_progress
            self._prev_seg_idx = current_seg

            reward += progress_delta * self._progress_scale

            if int(np.floor(self._cumulative_progress)) > prev_lap:
                reward += self._goal_reward

            info["progress"] = float(current_progress)

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.array(
            [self.state.x, self.state.y, self.state.heading, self.state.velocity],
            dtype=np.float64,
        )


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


def _progress_along_centerline(px, py, hw, hh, prev_seg_idx=None):
    total_perimeter = 4 * (hw + hh)
    seg_lengths = [2 * hw, 2 * hh, 2 * hw, 2 * hh]
    cum_length = np.float64(0.0)
    best_dist = np.inf
    best_progress = np.float64(0.0)
    best_seg_idx = 0

    for seg_idx, (x1, y1, x2, y2) in enumerate(_rectangle_edges(hw, hh)):
        sx = x2 - x1
        sy = y2 - y1
        seg_len_sq = sx * sx + sy * sy
        if seg_len_sq == 0.0:
            continue
        t = ((px - x1) * sx + (py - y1) * sy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        cx = x1 + t * sx
        cy = y1 + t * sy
        dx = px - cx
        dy = py - cy
        dist = np.sqrt(dx * dx + dy * dy)

        effective_dist = dist
        if prev_seg_idx is not None and seg_idx == prev_seg_idx:
            effective_dist -= 1e-10

        if effective_dist < best_dist:
            best_dist = dist
            best_progress = (cum_length + t * seg_lengths[seg_idx]) / total_perimeter
            best_seg_idx = seg_idx

        cum_length += seg_lengths[seg_idx]

    return (best_progress, best_seg_idx)
