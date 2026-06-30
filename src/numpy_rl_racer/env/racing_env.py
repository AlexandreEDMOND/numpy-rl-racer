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


class ProceduralTrack:
    def __init__(
        self,
        seed=0,
        radius=6.0,
        track_width=2.0,
        num_control_points=12,
        radial_noise=0.28,
        smoothing_steps=3,
    ):
        if num_control_points < 5:
            raise ValueError("num_control_points must be >= 5")
        if radius <= 0.0:
            raise ValueError("radius must be > 0")
        if track_width <= 0.0:
            raise ValueError("track_width must be > 0")
        if radial_noise < 0.0:
            raise ValueError("radial_noise must be >= 0")
        if smoothing_steps < 0:
            raise ValueError("smoothing_steps must be >= 0")

        self.seed = seed
        self.radius = np.float64(radius)
        self.track_width = np.float64(track_width)
        self.num_control_points = int(num_control_points)
        self.radial_noise = np.float64(radial_noise)
        self.smoothing_steps = int(smoothing_steps)

        rng = np.random.RandomState(seed)
        points = self._generate_control_points(rng)
        for _ in range(self.smoothing_steps):
            points = _chaikin_closed(points)
        self._set_centerline(points)

    @property
    def half_w(self):
        return np.float64(max(abs(self.x_min), abs(self.x_max)))

    @property
    def half_h(self):
        return np.float64(max(abs(self.y_min), abs(self.y_max)))

    @property
    def goal_position(self):
        x, y, _ = self.start_position
        return (x, y)

    @property
    def start_position(self):
        x, y = self.centerline_points[0]
        tangent = self._segment_angles[0]
        return (np.float64(x), np.float64(y), np.float64(tangent))

    def sample_centerline_point(self, rng=None):
        rng = np.random if rng is None else rng
        return self.get_centerline_point(rng.uniform(0.0, 1.0))

    def progress_along_centerline(self, x, y):
        _, _, _, progress = self._nearest_centerline_projection(x, y)
        return progress

    def is_on_track(self, x, y):
        dist, _, _, _ = self._nearest_centerline_projection(x, y)
        return dist <= self.track_width / np.float64(2.0)

    def get_centerline_point(self, progress):
        progress = np.float64(progress) % np.float64(1.0)
        target_len = progress * self._perimeter
        idx = int(np.searchsorted(self._cum_lengths[1:], target_len, side="right"))
        idx = min(idx, len(self._segments) - 1)
        seg_start_len = self._cum_lengths[idx]
        seg_len = self._segment_lengths[idx]
        t = np.float64(0.0) if seg_len == 0.0 else (target_len - seg_start_len) / seg_len
        point = self._seg_starts[idx] + t * self._segments[idx]
        angle = self._segment_angles[idx]
        return (np.float64(point[0]), np.float64(point[1]), np.float64(angle))

    def centerline_info(self, x, y):
        dist, _, angle, _ = self._nearest_centerline_projection(x, y)
        return (np.float64(dist), np.float64(angle))

    def _generate_control_points(self, rng):
        n = self.num_control_points
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        angle_step = 2.0 * np.pi / n
        jitter = rng.uniform(-0.25 * angle_step, 0.25 * angle_step, size=n)
        angles = base_angles + jitter - np.pi / 2.0
        radii = self.radius * (1.0 + rng.uniform(-self.radial_noise, self.radial_noise, size=n))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        start_idx = int(np.argmin(points[:, 1] + 0.15 * np.abs(points[:, 0])))
        return np.roll(points, -start_idx, axis=0).astype(np.float64)

    def _set_centerline(self, points):
        self.centerline_points = np.asarray(points, dtype=np.float64)
        self._seg_starts = self.centerline_points
        self._seg_ends = np.roll(self.centerline_points, -1, axis=0)
        self._segments = self._seg_ends - self._seg_starts
        self._segment_lengths = np.sqrt(np.sum(self._segments * self._segments, axis=1))
        if np.any(self._segment_lengths <= 1e-12):
            raise ValueError("generated track contains a degenerate segment")
        self._perimeter = np.float64(np.sum(self._segment_lengths))
        self._cum_lengths = np.concatenate([
            np.array([0.0], dtype=np.float64),
            np.cumsum(self._segment_lengths),
        ])
        self._segment_angles = np.arctan2(self._segments[:, 1], self._segments[:, 0])
        self._compute_boundaries()

    def _compute_boundaries(self):
        prev_angles = np.roll(self._segment_angles, 1)
        avg_x = np.cos(prev_angles) + np.cos(self._segment_angles)
        avg_y = np.sin(prev_angles) + np.sin(self._segment_angles)
        tangents = np.arctan2(avg_y, avg_x)
        normals = np.column_stack([-np.sin(tangents), np.cos(tangents)])
        half_tw = self.track_width / np.float64(2.0)

        self.left_boundary = self.centerline_points + normals * half_tw
        self.right_boundary = self.centerline_points - normals * half_tw
        self.outer_boundary = np.vstack([self.left_boundary, self.left_boundary[:1]])
        self.inner_boundary = np.vstack([self.right_boundary, self.right_boundary[:1]])

        left_segments = _segments_from_closed_points(self.left_boundary)
        right_segments = _segments_from_closed_points(self.right_boundary)
        self.boundary_segments = np.vstack([left_segments, right_segments]).astype(np.float64)

        all_points = np.vstack([self.left_boundary, self.right_boundary])
        margin = float(self.track_width)
        self.x_min = np.float64(np.min(all_points[:, 0]) - margin)
        self.x_max = np.float64(np.max(all_points[:, 0]) + margin)
        self.y_min = np.float64(np.min(all_points[:, 1]) - margin)
        self.y_max = np.float64(np.max(all_points[:, 1]) + margin)

    def _nearest_centerline_projection(self, x, y):
        p = np.array([x, y], dtype=np.float64)
        v = p - self._seg_starts
        seg_len_sq = self._segment_lengths * self._segment_lengths
        t = np.sum(v * self._segments, axis=1) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        projected = self._seg_starts + t[:, None] * self._segments
        delta = p - projected
        dist_sq = np.sum(delta * delta, axis=1)
        idx = int(np.argmin(dist_sq))
        distance = np.sqrt(dist_sq[idx])
        progress = (self._cum_lengths[idx] + t[idx] * self._segment_lengths[idx]) / self._perimeter
        if progress >= 1.0:
            progress = np.float64(0.0)
        return distance, projected[idx], self._segment_angles[idx], np.float64(progress)


class RacingEnv:
    def __init__(
        self,
        track_road_width=2.0,
        dt=0.1,
        track=None,
        track_seed=0,
        track_radius=6.0,
        track_points=12,
        track_variation=0.28,
        track_smoothing=3,
        randomize_start=True,
        obstacles=None,
        time_penalty=0.0,
        num_reward_lines=10,
        reward_line_reward=0.5,
        use_lidar=False,
        num_lidar_rays=8,
        lidar_max_range=10.0,
        observation_mode="state",
        reward_mode="legacy",
        local_ray_angles=None,
        progress_reward_scale=10.0,
        lap_bonus=5.0,
        off_track_penalty=5.0,
        collision_penalty=5.0,
        step_penalty=0.0,
    ):
        self.track = track if track is not None else ProceduralTrack(
            seed=track_seed,
            radius=track_radius,
            track_width=track_road_width,
            num_control_points=track_points,
            radial_noise=track_variation,
            smoothing_steps=track_smoothing,
        )
        if observation_mode not in ("state", "local"):
            raise ValueError(f"observation_mode must be 'state' or 'local', got {observation_mode!r}")
        if reward_mode not in ("legacy", "progress"):
            raise ValueError(f"reward_mode must be 'legacy' or 'progress', got {reward_mode!r}")
        self.randomize_start = randomize_start
        self.car = KinematicCar()
        self.dt = np.float64(dt)
        self.time_penalty = np.float64(time_penalty)
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.progress_reward_scale = np.float64(progress_reward_scale)
        self.lap_bonus = np.float64(lap_bonus)
        self.off_track_penalty = np.float64(off_track_penalty)
        self.collision_penalty = np.float64(collision_penalty)
        self.step_penalty = np.float64(step_penalty)
        self.rng = np.random.RandomState()
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
        if local_ray_angles is None:
            local_ray_angles = [np.pi / 2.0, np.pi / 4.0, 0.0, -np.pi / 4.0, -np.pi / 2.0]
        self.local_ray_angles = np.asarray(local_ray_angles, dtype=np.float64)
        self._lidar_boundary_segments = self._compute_lidar_boundary_segments()

    @property
    def observation_dim(self):
        if self.observation_mode == "local":
            return 4 + len(self.local_ray_angles)
        if self.use_lidar:
            return 6 + self.num_lidar_rays
        if self.obstacles:
            return 8
        return 6

    @property
    def goal_position(self):
        return self.track.goal_position

    def reset(self, seed=None, randomize_start=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        if randomize_start is None:
            randomize_start = self.randomize_start
        if randomize_start:
            cx, cy, tangent = self.track.sample_centerline_point(rng=self.rng)
            max_lateral = np.float64(0.2) * self.track.track_width
            lateral = self.rng.uniform(-max_lateral, max_lateral)
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

        progress_diff = self.current_progress - self.prev_progress
        lap_completed = progress_diff < -np.float64(0.5)
        if lap_completed:
            progress_diff += np.float64(1.0)

        if self.reward_mode == "progress":
            reward = self.progress_reward_scale * progress_diff - self.step_penalty
            if not on_track:
                reward -= self.off_track_penalty
            if obstacle_collision:
                reward -= self.collision_penalty
            if lap_completed:
                reward += self.lap_bonus
        else:
            reward = np.float64(0.1 if on_track else -1.0)
            if obstacle_collision:
                reward += np.float64(-1.0)

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

            reward += np.float64(0.5) * progress_diff

        if lap_completed:
            self.lap_count += 1
            if self.reward_mode == "legacy":
                reward += np.float64(1.0)
            self._collected_reward_lines[:] = False
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
        if not self.use_lidar and self.observation_mode != "local":
            return np.empty((0, 2, 2), dtype=np.float64)
        return np.asarray(self.track.boundary_segments, dtype=np.float64)

    def _lidar_readings(self):
        return self._ray_readings(self._lidar_angles)

    def _local_ray_readings(self):
        return self._ray_readings(self.local_ray_angles)

    def _ray_readings(self, relative_angles):
        origin = np.array([self.state.x, self.state.y], dtype=np.float64)
        heading = self.state.heading
        angles = relative_angles + heading
        dirs = np.column_stack([np.cos(angles), np.sin(angles)])

        num_rays = len(relative_angles)
        max_range = self.lidar_max_range
        min_dists = np.full(num_rays, max_range, dtype=np.float64)

        segments = self._lidar_boundary_segments
        if len(segments) > 0:
            for i in range(num_rays):
                d_vec = _ray_segments_distances_vector(origin, dirs[i], segments)
                if d_vec < min_dists[i]:
                    min_dists[i] = d_vec

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

        if self.observation_mode == "local":
            speed_norm = self.state.velocity / self.car.max_speed
            return np.array([
                speed_norm,
                np.sin(heading_error),
                np.cos(heading_error),
                dist_to_edge_normalized,
                *self._local_ray_readings().tolist(),
            ], dtype=np.float64)

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


def _chaikin_closed(points):
    p0 = np.asarray(points, dtype=np.float64)
    p1 = np.roll(p0, -1, axis=0)
    q = 0.75 * p0 + 0.25 * p1
    r = 0.25 * p0 + 0.75 * p1
    out = np.empty((len(points) * 2, 2), dtype=np.float64)
    out[0::2] = q
    out[1::2] = r
    return out


def _segments_from_closed_points(points):
    starts = np.asarray(points, dtype=np.float64)
    ends = np.roll(starts, -1, axis=0)
    return np.stack([starts, ends], axis=1)


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
