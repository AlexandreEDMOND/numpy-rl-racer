import numpy as np
import matplotlib.patches as mpatches
from matplotlib.path import Path


class MatplotlibRenderer:
    def __init__(self, track, figsize=(8, 6), headless=False, reward_line_progress=None, ax=None):
        self.track = track
        self._headless = headless
        self._owns_figure = ax is None
        if ax is not None:
            if not headless:
                import matplotlib.pyplot as plt
                self._plt = plt
            self.fig = ax.figure
            self.ax = ax
        elif headless:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=figsize)
            FigureCanvasAgg(self.fig)
            self.ax = self.fig.add_subplot(111)
        else:
            import matplotlib.pyplot as plt
            self._plt = plt
            self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect("equal")
        self._reward_line_endpoints = []
        if reward_line_progress is not None:
            from numpy_rl_racer.env.racing_env import reward_line_endpoints as _rle
            self._reward_line_endpoints = [_rle(self.track, p) for p in reward_line_progress]
        self._compute_boundary_lines()
        self._draw_background()
        self._recording = False
        self._recording_frames = []

    def _compute_boundary_lines(self):
        self._outer_boundary = np.asarray(self.track.outer_boundary, dtype=np.float64)
        self._inner_boundary = np.asarray(self.track.inner_boundary, dtype=np.float64)
        self._centerline = np.vstack([
            self.track.centerline_points,
            self.track.centerline_points[:1],
        ])

    def _draw_boundary_lines(self):
        self.ax.plot(self._outer_boundary[:, 0], self._outer_boundary[:, 1],
                     color="#666666", linewidth=0.8, zorder=2)
        self.ax.plot(self._inner_boundary[:, 0], self._inner_boundary[:, 1],
                     color="#666666", linewidth=0.8, zorder=2)

    def _draw_background(self):
        self._draw_track_background()
        self._draw_boundary_lines()
        self._draw_reward_lines()

    def _draw_track_background(self):
        outer = self._outer_boundary
        inner = self._inner_boundary
        vertices = np.vstack([outer, inner[::-1], outer[:1]])
        codes = (
            [Path.MOVETO] + [Path.LINETO] * (len(outer) - 1)
            + [Path.MOVETO] + [Path.LINETO] * (len(inner) - 1)
            + [Path.CLOSEPOLY]
        )
        path = Path(vertices, codes)
        self.ax.add_patch(
            mpatches.PathPatch(path, facecolor="#dddddd", edgecolor="#888888", linewidth=1)
        )
        self.ax.plot(
            self._centerline[:, 0],
            self._centerline[:, 1],
            "--",
            color="#aaaaaa",
            linewidth=0.5,
            zorder=1,
        )
        gx, gy = self.track.goal_position
        self.ax.plot(gx, gy, "*", color="#1a7c1a", markersize=10, zorder=3)

        self.ax.set_xlim(float(self.track.x_min), float(self.track.x_max))
        self.ax.set_ylim(float(self.track.y_min), float(self.track.y_max))
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("numpy-rl-racer")

    def _draw_reward_lines(self):
        for (x1, y1), (x2, y2) in self._reward_line_endpoints:
            self.ax.plot([x1, x2], [y1, y2], color="#2e86c1", linewidth=1.5,
                         linestyle="-", alpha=0.6, zorder=3)

    def start_recording(self):
        self._recording = True
        self._recording_frames = []

    def capture_frame(self):
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        self._recording_frames.append(rgba[:, :, :3].copy())

    def stop_recording(self, clear=True):
        self._recording = False
        if clear:
            self._recording_frames = []

    def save_animation(self, filepath, fps=10):
        if not self._recording_frames:
            raise RuntimeError(
                "No frames recorded. Call start_recording() and render() before save_animation()."
            )
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for GIF recording. Install it via: uv add pillow"
            )
        duration = int(1000 / fps)
        frames = [Image.fromarray(frame) for frame in self._recording_frames]
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

    def render(
        self,
        state,
        step=None,
        reward=None,
        obstacles=None,
        total_reward=None,
        lap_count=None,
        reward_lines_crossed=None,
    ):
        self.ax.clear()
        self._draw_background()

        if obstacles:
            for obs in obstacles:
                circle = mpatches.Circle(
                    (float(obs.x), float(obs.y)),
                    float(obs.radius),
                    facecolor="#e74c3c",
                    edgecolor="#c0392b",
                    linewidth=1.5,
                    zorder=4,
                )
                self.ax.add_patch(circle)

        self.ax.plot(float(state.x), float(state.y), "o", color="red", markersize=8, zorder=5)

        dx = np.cos(float(state.heading)) * 0.5
        dy = np.sin(float(state.heading)) * 0.5
        self.ax.arrow(
            float(state.x), float(state.y), dx, dy,
            head_width=0.2, head_length=0.2, fc="red", ec="red", zorder=6,
        )

        title = "numpy-rl-racer"
        if step is not None:
            title += f"   step={step}"
        if reward is not None:
            title += f"   step_reward={reward:.2f}"
        if total_reward is not None:
            title += f"   total={total_reward:.2f}"
        if lap_count is not None:
            title += f"   laps={lap_count}"
        if reward_lines_crossed is not None:
            title += f"   lines={reward_lines_crossed}"
        self.ax.set_title(title)

        self.fig.canvas.draw_idle()
        if self._recording:
            self.capture_frame()
        if not self._headless:
            import matplotlib.pyplot as plt
            plt.pause(0.01)

    def show(self):
        if self._headless:
            print("[Headless mode] show() is a no-op")
        else:
            self._plt.show(block=True)

    def close(self):
        if not self._headless and self._owns_figure:
            self._plt.close(self.fig)
        if self._owns_figure:
            self.fig.clear()
        self._recording = False
        self._recording_frames = []
