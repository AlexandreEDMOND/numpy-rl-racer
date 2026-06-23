import numpy as np
from matplotlib.path import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class MatplotlibRenderer:
    def __init__(self, track, figsize=(8, 6)):
        self.track = track
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect("equal")
        self._draw_background()

    def _draw_background(self):
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

    def render(self, state, step=None, reward=None):
        self.ax.clear()
        self._draw_background()

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
        plt.pause(0.01)

    def show(self):
        plt.show(block=True)

    def close(self):
        plt.close(self.fig)
