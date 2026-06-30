"""Generate an annotated procedural environment overview image for the README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from numpy_rl_racer.env.racing_env import ProceduralTrack, reward_line_endpoints
from numpy_rl_racer.rendering.matplotlib_renderer import MatplotlibRenderer


def _reward_line_progress():
    return list(np.linspace(0.0, 1.0, 12)[1:-1])


def _draw_background(ax, track):
    renderer = MatplotlibRenderer.__new__(MatplotlibRenderer)
    renderer.track = track
    renderer.fig = ax.figure
    renderer.ax = ax
    renderer._headless = True
    renderer._reward_line_endpoints = [
        reward_line_endpoints(track, p) for p in _reward_line_progress()
    ]
    renderer._compute_boundary_lines()
    renderer._draw_background()


def _place_car(ax, track, progress):
    x, y, heading = track.get_centerline_point(progress)
    ax.plot(x, y, "o", color="red", markersize=10, zorder=7)
    dx = np.cos(heading) * 0.6
    dy = np.sin(heading) * 0.6
    ax.arrow(x, y, dx, dy, head_width=0.25, head_length=0.25,
             fc="red", ec="red", zorder=8)


def draw_track(ax, seed, progress):
    track = ProceduralTrack(seed=seed, radius=6.0, track_width=2.0)
    _draw_background(ax, track)
    _place_car(ax, track, progress)
    gx, gy = track.goal_position
    ax.plot(gx, gy, "*", color="#1a7c1a", markersize=14, zorder=6,
            markeredgecolor="#0d4f0d", markeredgewidth=0.5)
    ax.set_title(f"Procedural seed {seed}", fontsize=12, fontweight="bold")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, seed, progress in zip(axes, [0, 7, 21], [0.12, 0.42, 0.72]):
        draw_track(ax, seed, progress)
        ax.set_xlabel("X position", fontsize=10)
        ax.set_ylabel("Y position", fontsize=10)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="Car (position)"),
        Line2D([0], [0], color="red", linewidth=2,
               label="Car heading (direction)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1a7c1a",
               markersize=12, label="Start / finish"),
        Patch(facecolor="#dddddd", edgecolor="#888888", label="Road surface"),
        Line2D([0], [0], color="#666666", linewidth=0.8, label="Road boundaries"),
        Line2D([0], [0], linestyle="--", color="#aaaaaa", linewidth=0.5,
               label="Centerline"),
        Line2D([0], [0], color="#2e86c1", linewidth=1.5,
               label="Reward line"),
    ]

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=7, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("NumPy RL Racer - Procedural Track Overview",
                 fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig("images/environment_overview.png", dpi=150,
                bbox_inches="tight", pad_inches=0.3)
    print("Saved images/environment_overview.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
