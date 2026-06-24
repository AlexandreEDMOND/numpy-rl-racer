import numpy as np
from matplotlib.path import Path
import matplotlib.patches as mpatches


class MatplotlibRenderer:
    def __init__(self, track, figsize=(8, 6), headless=False):
        self.track = track
        self._headless = headless
        if headless:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self.fig = Figure(figsize=figsize)
            FigureCanvasAgg(self.fig)
            self.ax = self.fig.add_subplot(111)
        else:
            import matplotlib.pyplot as plt
            self._plt = plt
            self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect("equal")
        self._compute_boundary_lines()
        self._draw_background()
        self._recording = False
        self._recording_frames = []

    def _compute_boundary_lines(self):
        if hasattr(self.track, 'radius'):
            R = float(self.track.radius)
            tw2 = float(self.track.track_width) / 2.0
            theta = np.linspace(0, 2 * np.pi, 200)
            self._outer_boundary = np.column_stack([
                (R + tw2) * np.cos(theta),
                (R + tw2) * np.sin(theta),
            ])
            self._inner_boundary = np.column_stack([
                (R - tw2) * np.cos(theta),
                (R - tw2) * np.sin(theta),
            ])
        else:
            hw = float(self.track.half_w)
            hh = float(self.track.half_h)
            tw2 = float(self.track.track_width) / 2.0
            self._outer_boundary = np.array([
                [-hw - tw2, -hh - tw2],
                [hw + tw2, -hh - tw2],
                [hw + tw2, hh + tw2],
                [-hw - tw2, hh + tw2],
                [-hw - tw2, -hh - tw2],
            ])
            self._inner_boundary = np.array([
                [-hw + tw2, -hh + tw2],
                [hw - tw2, -hh + tw2],
                [hw - tw2, hh - tw2],
                [-hw + tw2, hh - tw2],
                [-hw + tw2, -hh + tw2],
            ])

    def _draw_boundary_lines(self):
        self.ax.plot(self._outer_boundary[:, 0], self._outer_boundary[:, 1],
                      color="#666666", linewidth=0.8, zorder=2)
        self.ax.plot(self._inner_boundary[:, 0], self._inner_boundary[:, 1],
                      color="#666666", linewidth=0.8, zorder=2)

    def _draw_background(self):
        if hasattr(self.track, 'radius'):
            self._draw_circular_background()
        else:
            self._draw_rectangular_background()
        self._draw_boundary_lines()

    def _draw_rectangular_background(self):
        hw = float(self.track.half_w)
        hh = float(self.track.half_h)
        tw = float(self.track.track_width)
        tw2 = tw / 2.0

        outer = [(-hw - tw2, -hh - tw2), (hw + tw2, -hh - tw2), (hw + tw2, hh + tw2), (-hw - tw2, hh + tw2)]
        inner = [(-hw + tw2, -hh + tw2), (hw - tw2, -hh + tw2), (hw - tw2, hh - tw2), (-hw + tw2, hh - tw2)]

        path = Path(
            outer + inner[::-1] + outer[:1],
            [
                Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
                Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
                Path.CLOSEPOLY,
            ],
        )
        self.ax.add_patch(mpatches.PathPatch(path, facecolor="#dddddd", edgecolor="#888888", linewidth=1))

        edges = [
            (-hw, -hh, hw, -hh),
            (hw, -hh, hw, hh),
            (hw, hh, -hw, hh),
            (-hw, hh, -hw, -hh),
        ]
        for x1, y1, x2, y2 in edges:
            self.ax.plot([x1, x2], [y1, y2], "--", color="#aaaaaa", linewidth=0.5, zorder=1)

        margin = tw
        self.ax.set_xlim(-hw - margin, hw + margin)
        self.ax.set_ylim(-hh - margin, hh + margin)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("numpy-rl-racer")

    def _draw_circular_background(self):
        R = float(self.track.radius)
        tw = float(self.track.track_width)
        tw2 = tw / 2.0

        self.ax.add_patch(
            mpatches.Circle((0, 0), R + tw2, facecolor="#dddddd", edgecolor="#888888", linewidth=1)
        )
        self.ax.add_patch(
            mpatches.Circle((0, 0), R - tw2, facecolor=self.fig.get_facecolor(), edgecolor="#888888", linewidth=1)
        )
        self.ax.add_patch(
            mpatches.Circle((0, 0), R, fill=False, linestyle="--", color="#aaaaaa", linewidth=0.5)
        )

        margin = tw
        self.ax.set_xlim(-R - margin, R + margin)
        self.ax.set_ylim(-R - margin, R + margin)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("numpy-rl-racer")

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
        from PIL import Image
        duration = int(1000 / fps)
        frames = [Image.fromarray(frame) for frame in self._recording_frames]
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

    def render(self, state, step=None, reward=None, obstacles=None):
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
            title += f"   reward={reward:.2f}"
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
        if not self._headless:
            self._plt.close(self.fig)
        self.fig.clear()
        self._recording = False
        self._recording_frames = []
