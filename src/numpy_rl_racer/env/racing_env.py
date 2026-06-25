from dataclasses import dataclass

import numpy as np

from .car import CarState, KinematicCar
from .utils import normalize_angle


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float = 0.5


CAR_COLLISION_RADIUS = 0.3


class RectangularTrack:
    def __init__(self, width=10.0, height=8.0, track_width=2.0):
        self.half_w = np.float64(width / 2.0)
        self.half_h = np.float64(height / 2.0)
        self.track_width = np.float64(track_width)
        self._perimeter = 4.0 * (self.half_w + self.half_h)

    @property
    def goal_position(self):
        return (np.float64(0.0), -self.half_h)

    @property
    def start_position(self):
        return (np.float64(0.0), -self.half_h, np.float64(0.0))

    def sample_centerline_point(self):
        hw, hh = self.half_w, self.half_h
        r = np.random.uniform(0.0, self._perimeter)
        cum = np.float64(0.0)
        for x1, y1, x2, y2 in _centerline_edges(hw, hh):
            sx = x2 - x1
            sy = y2 - y1
            seg_len = np.sqrt(sx * sx + sy * sy)
            cum_next = cum + seg_len
            if r <= cum_next:
                t = np.float64(0.0) if seg_len == 0.0 else (r - cum) / seg_len
                cx = x1 + t * sx
                cy = y1 + t * sy
                angle = np.arctan2(sy, sx)
                return (cx, cy, angle)
            cum = cum_next
        return self.start_position

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

    def get_centerline_point(self, progress):
        target_len = progress * self._perimeter
        cum_len = np.float64(0.0)
        for x1, y1, x2, y2 in _centerline_edges(self.half_w, self.half_h):
            sx = x2 - x1
            sy = y2 - y1
            seg_len = np.sqrt(sx * sx + sy * sy)
            cum_next = cum_len + seg_len
            if target_len <= cum_next:
                t = np.float64(0.0) if seg_len == 0.0 else (target_len - cum_len) / seg_len
                cx = x1 + t * sx
                cy = y1 + t * sy
                angle = np.arctan2(sy, sx)
                return (np.float64(cx), np.float64(cy), np.float64(angle))
            cum_len = cum_next
        return self.start_position

    def centerline_info(self, x, y):
        px, py = np.float64(x), np.float64(y)
        hw, hh = self.half_w, self.half_h

        best_dist = np.inf
        best_angle = np.float64(0.0)

        for x1, y1, x2, y2 in _centerline_edges(hw, hh):
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
            angle = np.arctan2(sy, sx)
            if dist < best_dist:
                best_dist = dist
                best_angle = angle

        return best_dist, best_angle


class CircularTrack:
    def __init__(self, radius=6.0, track_width=2.0):
        self.radius = np.float64(radius)
        self.track_width = np.float64(track_width)
        self._perimeter = np.float64(2.0 * np.pi * radius)

    @property
    def half_w(self):
        return self.radius

    @property
    def half_h(self):
        return self.radius

    @property
    def goal_position(self):
        return (np.float64(0.0), -self.radius)

    @property
    def start_position(self):
        return (np.float64(0.0), -self.radius, np.float64(0.0))

    def sample_centerline_point(self):
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        x = self.radius * np.sin(theta)
        y = -self.radius * np.cos(theta)
        return (x, y, theta)

    def progress_along_centerline(self, x, y):
        px, py = np.float64(x), np.float64(y)
        angle = np.arctan2(px, -py)
        if angle < 0:
            angle += 2.0 * np.pi
        return angle / (2.0 * np.pi)

    def is_on_track(self, x, y):
        px, py = np.float64(x), np.float64(y)
        dist = np.sqrt(px * px + py * py)
        tw2 = self.track_width / 2.0
        return (self.radius - tw2) <= dist <= (self.radius + tw2)

    def get_centerline_point(self, progress):
        theta = progress * np.float64(2.0 * np.pi)
        x = self.radius * np.sin(theta)
        y = -self.radius * np.cos(theta)
        return (np.float64(x), np.float64(y), np.float64(theta))

    def centerline_info(self, x, y):
        px, py = np.float64(x), np.float64(y)
        dist = np.sqrt(px * px + py * py)
        dist_to_centerline = np.abs(dist - self.radius)
        angle = np.arctan2(px, -py)
        return dist_to_centerline, angle


class BezierTrack:
    def __init__(self, num_anchors=8, track_width=2.0, radius=6.0, seed=None):
        self.num_anchors = max(3, num_anchors)
        self.track_width = np.float64(track_width)
        self.radius = np.float64(radius)

        self._rng = np.random.RandomState(seed)

        # Generate anchor points at random angles and radii
        angles = np.sort(self._rng.uniform(0, 2 * np.pi, self.num_anchors))
        radii = self._rng.uniform(0.5 * radius, 1.5 * radius, self.num_anchors)
        anchors = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        # Compute cubic Bezier control points (Catmull-Rom -> Bezier)
        n = self.num_anchors
        cp_list = []
        for i in range(n):
            p0 = anchors[i]
            p3 = anchors[(i + 1) % n]
            p_prev = anchors[(i - 1) % n]
            p_next = anchors[(i + 2) % n]
            p1 = p0 + (p3 - p_prev) / 6.0
            p2 = p3 - (p_next - p0) / 6.0
            cp_list.extend([p0, p1, p2])
        self._control_points = np.array(cp_list)
        self._anchors = anchors

        self._precompute()

    def _precompute(self):
        n_seg = self.num_anchors
        samples_per_seg = 2000 // n_seg
        cp = self._control_points

        pts_list = []
        for i in range(n_seg):
            P0 = cp[3 * i]
            P1 = cp[3 * i + 1]
            P2 = cp[3 * i + 2]
            P3 = cp[3 * ((i + 1) % n_seg)]

            t = np.linspace(0, 1, samples_per_seg, endpoint=(i == n_seg - 1))
            t2 = t * t
            t3 = t2 * t
            mt = 1.0 - t
            mt2 = mt * mt
            mt3 = mt2 * mt

            pts = (mt3[:, None] * P0 + 3.0 * mt2[:, None] * t[:, None] * P1
                   + 3.0 * mt[:, None] * t2[:, None] * P2 + t3[:, None] * P3)
            pts_list.append(pts)

        self._centerline = np.vstack(pts_list)
        self._n = len(self._centerline)
        self._cs_x = self._centerline[:, 0]
        self._cs_y = self._centerline[:, 1]

        # Cumulative distances along centerline
        diffs = np.diff(self._centerline, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        self._cumulative_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._perimeter = self._cumulative_dist[-1]

        # Tangents via central differences (with wrap-around)
        tangents = np.zeros(self._n)
        for i in range(self._n):
            i_prev = (i - 1) % self._n
            i_next = (i + 1) % self._n
            dx = self._cs_x[i_next] - self._cs_x[i_prev]
            dy = self._cs_y[i_next] - self._cs_y[i_prev]
            tangents[i] = np.arctan2(dy, dx)
        self._cs_tangents = tangents

    @property
    def goal_position(self):
        return (np.float64(self._cs_x[0]), np.float64(self._cs_y[0]))

    @property
    def start_position(self):
        return (np.float64(self._cs_x[0]), np.float64(self._cs_y[0]), np.float64(self._cs_tangents[0]))

    @property
    def half_w(self):
        return np.float64(np.max(np.abs(self._cs_x)) + float(self.track_width))

    @property
    def half_h(self):
        return np.float64(np.max(np.abs(self._cs_y)) + float(self.track_width))

    def sample_centerline_point(self, t=None):
        if t is None:
            t = self._rng.uniform(0.0, 1.0)
        return self.get_centerline_point(t)

    def progress_along_centerline(self, x, y):
        px, py = np.float64(x), np.float64(y)
        dx = self._cs_x - px
        dy = self._cs_y - py
        dist_sq = dx * dx + dy * dy
        best_idx = np.argmin(dist_sq)
        return np.float64(self._cumulative_dist[best_idx] / self._perimeter)

    def is_on_track(self, x, y):
        px, py = np.float64(x), np.float64(y)
        dx = self._cs_x - px
        dy = self._cs_y - py
        dist_sq = dx * dx + dy * dy
        min_dist = np.sqrt(np.min(dist_sq))
        return min_dist <= self.track_width / np.float64(2.0)

    def get_centerline_point(self, progress):
        target_len = np.clip(progress, 0.0, 1.0) * self._perimeter
        idx = np.searchsorted(self._cumulative_dist, target_len)
        idx = np.clip(idx, 0, self._n - 1)
        return (np.float64(self._cs_x[idx]), np.float64(self._cs_y[idx]), np.float64(self._cs_tangents[idx]))

    def centerline_info(self, x, y):
        px, py = np.float64(x), np.float64(y)
        dx = self._cs_x - px
        dy = self._cs_y - py
        dist_sq = dx * dx + dy * dy
        best_idx = np.argmin(dist_sq)
        dist = np.sqrt(dist_sq[best_idx])
        tangent = self._cs_tangents[best_idx]
        return (dist, tangent)


class Figure8Track:
    def __init__(self, radius=6.0, track_width=2.0):
        self.radius = np.float64(radius)
        self.track_width = np.float64(track_width)
        self._n = 2000
        self._theta_offset = 7.0 * np.pi / 4.0
        self._precompute()

    def _precompute(self):
        n = self._n
        self._t_vals = np.linspace(0.0, 1.0, n, endpoint=False)
        thetas = 2.0 * np.pi * self._t_vals + self._theta_offset
        ct = np.cos(thetas)
        st = np.sin(thetas)
        self._cs_x = self.radius * ct
        self._cs_y = self.radius * st * ct
        self._cs_tangents = np.arctan2(np.cos(2.0 * thetas), -np.sin(thetas))

    @property
    def goal_position(self):
        theta = self._theta_offset
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta) * np.cos(theta)
        return (np.float64(x), np.float64(y))

    @property
    def start_position(self):
        gx, gy = self.goal_position
        return (gx, gy, np.float64(0.0))

    @property
    def half_w(self):
        return self.radius

    @property
    def half_h(self):
        return self.radius * np.float64(0.5)

    def sample_centerline_point(self, t=None):
        if t is None:
            t = np.random.uniform(0.0, 1.0)
        theta = 2.0 * np.pi * t + self._theta_offset
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta) * np.cos(theta)
        tangent = np.arctan2(np.cos(2.0 * theta), -np.sin(theta))
        return (np.float64(x), np.float64(y), np.float64(tangent))

    def progress_along_centerline(self, x, y):
        px, py = np.float64(x), np.float64(y)
        dx = self._cs_x - px
        dy = self._cs_y - py
        dist_sq = dx * dx + dy * dy
        best_idx = np.argmin(dist_sq)
        return self._t_vals[best_idx]

    def is_on_track(self, x, y):
        px, py = np.float64(x), np.float64(y)
        dx = self._cs_x - px
        dy = self._cs_y - py
        dist_sq = dx * dx + dy * dy
        min_dist = np.sqrt(np.min(dist_sq))
        tw2 = self.track_width / np.float64(2.0)
        return min_dist <= tw2

    def get_centerline_point(self, progress):
        return self.sample_centerline_point(t=np.clip(progress, 0.0, 1.0))

    def centerline_info(self, x, y):
        px, py = np.float64(x), np.float64(y)
        dx = self._cs_x - px
        dy = self._cs_y - py
        dist_sq = dx * dx + dy * dy
        best_idx = np.argmin(dist_sq)
        dist = np.sqrt(dist_sq[best_idx])
        tangent = self._cs_tangents[best_idx]
        return (dist, tangent)


class RacingEnv:
    def __init__(self, track_width=10.0, track_height=8.0, track_road_width=2.0, dt=0.1, track=None, track_type='rectangular', randomize_start=True, obstacles=None, time_penalty=0.0, num_reward_lines=10, reward_line_reward=0.5, use_lidar=False, num_lidar_rays=8, lidar_max_range=10.0):
        if track is not None:
            self.track = track
        elif track_type == 'figure8':
            radius = min(track_width, track_height) / 2.0
            self.track = Figure8Track(radius, track_road_width)
        elif track_type == 'bezier':
            self.track = BezierTrack(
                track_width=track_road_width,
                radius=min(track_width, track_height) / 2.0,
            )
        else:
            self.track = RectangularTrack(track_width, track_height, track_road_width)
        self.randomize_start = randomize_start
        self.car = KinematicCar()
        self.dt = np.float64(dt)
        self.time_penalty = np.float64(time_penalty)
        self.state = None
        self.current_progress = np.float64(0.0)
        self.prev_progress = np.float64(0.0)
        self.lap_count = 0
        self.elapsed_time = np.float64(0.0)
        self.obstacles = obstacles if obstacles is not None else []
        self.num_reward_lines = num_reward_lines
        self.reward_line_reward = np.float64(reward_line_reward)
        self._reward_line_progress = []
        self._collected_reward_lines = np.array([], dtype=bool)
        if num_reward_lines > 0:
            self._reward_line_progress = list(np.linspace(
                np.float64(0.0), np.float64(1.0), num_reward_lines + 2
            )[1:-1])
            self._collected_reward_lines = np.zeros(num_reward_lines, dtype=bool)

        self.use_lidar = use_lidar
        self.num_lidar_rays = num_lidar_rays
        self.lidar_max_range = np.float64(lidar_max_range)
        self._lidar_angles = np.linspace(0, 2 * np.pi, num_lidar_rays, endpoint=False).astype(np.float64)
        self._lidar_boundary_segments = self._compute_lidar_boundary_segments()

    @property
    def goal_position(self):
        return self.track.goal_position

    def reset(self, seed=None, randomize_start=None):
        if seed is not None:
            np.random.seed(seed)
        if randomize_start is None:
            randomize_start = self.randomize_start
        if randomize_start:
            cx, cy, tangent = self.track.sample_centerline_point()
            max_lateral = np.float64(0.2) * self.track.track_width
            lateral = np.random.uniform(-max_lateral, max_lateral)
            perp_angle = tangent + np.pi / np.float64(2.0)
            sx = cx + lateral * np.cos(perp_angle)
            sy = cy + lateral * np.sin(perp_angle)
            sheading = tangent
            self.current_progress = self.track.progress_along_centerline(sx, sy)
        else:
            sx, sy, sheading = self.track.start_position
            self.current_progress = np.float64(0.0)
        self.state = CarState(x=sx, y=sy, heading=sheading, velocity=0.0)
        self.prev_progress = np.float64(0.0)
        self.lap_count = 0
        self.elapsed_time = np.float64(0.0)
        self._collected_reward_lines[:] = False
        return self._get_observation()

    def step(self, action):
        steering, acceleration = np.float64(action[0]), np.float64(action[1])
        self.state = self.car.step(self.state, steering, acceleration, self.dt)
        on_track = self.track.is_on_track(self.state.x, self.state.y)
        done = not on_track

        obstacle_collision = self._check_obstacle_collision()
        if obstacle_collision:
            done = True

        self.elapsed_time += self.dt

        self.prev_progress = self.current_progress
        self.current_progress = self.track.progress_along_centerline(self.state.x, self.state.y)

        reward = np.float64(0.1 if on_track else -1.0)
        if obstacle_collision:
            reward += np.float64(-1.0)

        progress_diff = self.current_progress - self.prev_progress
        lap_completed = progress_diff < -np.float64(0.5)

        # Check reward line crossings
        for i, lp in enumerate(self._reward_line_progress):
            if self._collected_reward_lines[i]:
                continue
            crossed = False
            if self.prev_progress <= lp < self.current_progress:
                crossed = True
            elif lap_completed and (lp >= self.prev_progress or lp <= self.current_progress):
                crossed = True
            if crossed:
                reward += self.reward_line_reward
                self._collected_reward_lines[i] = True

        if lap_completed:
            progress_diff += np.float64(1.0)
            self.lap_count += 1
            reward += np.float64(1.0)
            self._collected_reward_lines[:] = False
        reward += np.float64(0.5) * progress_diff

        reward -= self.time_penalty * self.dt

        info = {
            'progress': self.current_progress,
            'lap_count': self.lap_count,
            'goal_position': self.goal_position,
            'elapsed_time': self.elapsed_time,
            'reward_lines_crossed': int(np.sum(self._collected_reward_lines)),
        }

        return self._get_observation(), reward, done, info

    def _check_obstacle_collision(self):
        for obs in self.obstacles:
            dx = self.state.x - obs.x
            dy = self.state.y - obs.y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < CAR_COLLISION_RADIUS + obs.radius:
                return True
        return False

    def _compute_lidar_boundary_segments(self):
        if not self.use_lidar:
            return np.empty((0, 2, 2), dtype=np.float64)

        segments = []

        if isinstance(self.track, RectangularTrack):
            hw, hh = self.track.half_w, self.track.half_h
            edges = _centerline_edges(hw, hh)
            tw2 = self.track.track_width / np.float64(2.0)

            left_ends = []
            right_ends = []
            for (x1, y1, x2, y2) in edges:
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx * dx + dy * dy)
                if length == 0:
                    continue
                px = -dy / length
                py = dx / length
                ls = (x1 + px * tw2, y1 + py * tw2)
                le = (x2 + px * tw2, y2 + py * tw2)
                rs = (x1 - px * tw2, y1 - py * tw2)
                re = (x2 - px * tw2, y2 - py * tw2)
                left_ends.append((ls, le))
                right_ends.append((rs, re))
                segments.append((ls, le))
                segments.append((rs, re))

            n = len(edges)
            for i in range(n):
                j = (i + 1) % n
                segments.append((right_ends[i][1], right_ends[j][0]))
                segments.append((left_ends[i][1], left_ends[j][0]))

        elif isinstance(self.track, Figure8Track):
            n = self.track._n
            cs_x = self.track._cs_x
            cs_y = self.track._cs_y
            tangents = self.track._cs_tangents
            tw2 = self.track.track_width / np.float64(2.0)

            px_arr = -np.sin(tangents) * tw2
            py_arr = np.cos(tangents) * tw2
            left_x = cs_x + px_arr
            left_y = cs_y + py_arr
            right_x = cs_x - px_arr
            right_y = cs_y - py_arr

            for i in range(n - 1):
                segments.append(((left_x[i], left_y[i]), (left_x[i + 1], left_y[i + 1])))
                segments.append(((right_x[i], right_y[i]), (right_x[i + 1], right_y[i + 1])))

        elif isinstance(self.track, BezierTrack):
            n = self.track._n
            cs_x = self.track._cs_x
            cs_y = self.track._cs_y
            tangents = self.track._cs_tangents
            tw2 = self.track.track_width / np.float64(2.0)

            px_arr = -np.sin(tangents) * tw2
            py_arr = np.cos(tangents) * tw2
            left_x = cs_x + px_arr
            left_y = cs_y + py_arr
            right_x = cs_x - px_arr
            right_y = cs_y - py_arr

            for i in range(n - 1):
                segments.append(((left_x[i], left_y[i]), (left_x[i + 1], left_y[i + 1])))
                segments.append(((right_x[i], right_y[i]), (right_x[i + 1], right_y[i + 1])))

        return np.array(segments, dtype=np.float64)

    def _lidar_readings(self):
        origin = np.array([self.state.x, self.state.y], dtype=np.float64)
        heading = self.state.heading
        angles = self._lidar_angles + heading
        dirs = np.column_stack([np.cos(angles), np.sin(angles)])

        num_rays = self.num_lidar_rays
        max_range = self.lidar_max_range
        min_dists = np.full(num_rays, max_range, dtype=np.float64)

        # Track boundary intersection
        if isinstance(self.track, CircularTrack):
            R = self.track.radius
            tw2 = self.track.track_width / np.float64(2.0)
            for cr in [R - tw2, R + tw2]:
                if cr > 0:
                    for i in range(num_rays):
                        d = _ray_circle_intersection_distance(origin, dirs[i], np.zeros(2, dtype=np.float64), cr)
                        if d < min_dists[i]:
                            min_dists[i] = d
        else:
            segments = self._lidar_boundary_segments
            n_seg = len(segments)
            if n_seg > 0:
                for i in range(num_rays):
                    d_vec = _ray_segments_distances_vector(origin, dirs[i], segments)
                    if d_vec < min_dists[i]:
                        min_dists[i] = d_vec

        # Obstacle intersection
        for obs in self.obstacles:
            center = np.array([obs.x, obs.y], dtype=np.float64)
            r = obs.radius
            for i in range(num_rays):
                d = _ray_circle_intersection_distance(origin, dirs[i], center, r)
                if d < min_dists[i]:
                    min_dists[i] = d

        return np.clip(min_dists / max_range, np.float64(0.0), np.float64(1.0))

    def _get_observation(self):
        dist_to_centerline, tangent_angle = self.track.centerline_info(
            self.state.x, self.state.y
        )
        half_tw = self.track.track_width / np.float64(2.0)
        dist_to_edge = half_tw - dist_to_centerline
        dist_to_edge_normalized = np.clip(
            dist_to_edge / half_tw, np.float64(0.0), np.float64(1.0)
        )
        heading_error = normalize_angle(self.state.heading - tangent_angle)

        obs = [
            self.state.x,
            self.state.y,
            self.state.heading,
            self.state.velocity,
            dist_to_edge_normalized,
            heading_error,
        ]

        if self.use_lidar:
            obs.extend(self._lidar_readings().tolist())
        elif self.obstacles:
            nearest_dist = np.inf
            nearest_angle = np.float64(0.0)
            for obs_obj in self.obstacles:
                dx = obs_obj.x - self.state.x
                dy = obs_obj.y - self.state.y
                dist = np.sqrt(dx * dx + dy * dy)
                angle = normalize_angle(np.arctan2(dy, dx) - self.state.heading)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_angle = angle
            max_dist = np.float64(
                self.track.half_w + self.track.half_h + self.track.track_width
            )
            normalized_dist = np.clip(nearest_dist / max_dist, 0.0, 1.0)
            normalized_angle = nearest_angle / np.pi
            obs.extend([normalized_dist, normalized_angle])

        return np.array(obs, dtype=np.float64)


def reward_line_endpoints(track, progress):
    cx, cy, tangent = track.get_centerline_point(progress)
    perp = tangent + np.pi / np.float64(2.0)
    half_tw = track.track_width / np.float64(2.0)
    x1 = cx - half_tw * np.cos(perp)
    y1 = cy - half_tw * np.sin(perp)
    x2 = cx + half_tw * np.cos(perp)
    y2 = cy + half_tw * np.sin(perp)
    return ((np.float64(x1), np.float64(y1)), (np.float64(x2), np.float64(y2)))


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


def _default_obstacles(track):
    if hasattr(track, 'radius'):
        R = float(track.radius)
        obstacles = [
            Obstacle(0.0, 0.0, 0.5),
            Obstacle(R * 0.35, R * 0.35, 0.45),
            Obstacle(-R * 0.35, -R * 0.35, 0.4),
        ]
    else:
        hw = float(track.half_w)
        hh = float(track.half_h)
        inner_hw = hw - track.track_width
        inner_hh = hh - track.track_width
        obstacles = [
            Obstacle(0.0, 0.0, 0.5),
            Obstacle(inner_hw * 0.6, -inner_hh * 0.4, 0.45),
            Obstacle(-inner_hw * 0.5, inner_hh * 0.5, 0.4),
        ]
    return obstacles


def _ray_segment_intersection_distance(origin, direction, p1, p2):
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    seg_dir = b - a
    cross_d_seg = d[0] * seg_dir[1] - d[1] * seg_dir[0]
    if abs(cross_d_seg) < 1e-12:
        return np.inf
    w = o - a
    t = (w[0] * seg_dir[1] - w[1] * seg_dir[0]) / cross_d_seg
    if t <= 1e-12:
        return np.inf
    s = (d[0] * w[1] - d[1] * w[0]) / cross_d_seg
    if s < 0.0 or s > 1.0:
        return np.inf
    return t


def _ray_segments_distances_vector(origin, direction, segments):
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    a = np.asarray(segments[:, 0, :], dtype=np.float64)
    b = np.asarray(segments[:, 1, :], dtype=np.float64)
    seg_dirs = b - a
    cross_d_seg = d[0] * seg_dirs[:, 1] - d[1] * seg_dirs[:, 0]
    parallel = np.abs(cross_d_seg) < 1e-12
    w = o - a
    t = np.full(len(segments), np.inf, dtype=np.float64)
    s = np.full(len(segments), -1.0, dtype=np.float64)
    nz = ~parallel
    if np.any(nz):
        t[nz] = (w[nz, 0] * seg_dirs[nz, 1] - w[nz, 1] * seg_dirs[nz, 0]) / cross_d_seg[nz]
        s[nz] = (d[0] * w[nz, 1] - d[1] * w[nz, 0]) / cross_d_seg[nz]
    valid = nz & (t > 1e-12) & (s >= 0.0) & (s <= 1.0)
    if np.any(valid):
        return np.min(t[valid])
    return np.inf


def _ray_circle_intersection_distance(origin, direction, center, radius):
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    e = o - c
    ed = e[0] * d[0] + e[1] * d[1]
    e_sq = e[0] * e[0] + e[1] * e[1]
    r_sq = radius * radius
    disc = ed * ed - e_sq + r_sq
    if disc < 0.0:
        return np.inf
    sqrt_disc = np.sqrt(disc)
    t1 = -ed - sqrt_disc
    t2 = -ed + sqrt_disc
    if t1 > 1e-12:
        return t1
    if t2 > 1e-12:
        return t2
    return np.inf


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
