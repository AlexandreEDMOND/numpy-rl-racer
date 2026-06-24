"""Generate an annotated environment overview image for the README."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from numpy_rl_racer.env.racing_env import (
    RectangularTrack,
    CircularTrack,
    Figure8Track,
)
from numpy_rl_racer.rendering.matplotlib_renderer import MatplotlibRenderer


def _draw_background(ax, track):
    renderer = MatplotlibRenderer.__new__(MatplotlibRenderer)
    renderer.track = track
    renderer.fig = ax.figure
    renderer.ax = ax
    renderer._headless = True
    renderer._compute_boundary_lines()
    renderer._draw_background()


def _place_car(ax, x, y, heading):
    ax.plot(x, y, "o", color="red", markersize=10, zorder=7)
    dx = np.cos(heading) * 0.6
    dy = np.sin(heading) * 0.6
    ax.arrow(x, y, dx, dy, head_width=0.25, head_length=0.25,
             fc="red", ec="red", zorder=8)


def _mark_goal(ax, x, y):
    ax.plot(x, y, "*", color="#1a7c1a", markersize=14, zorder=6,
            markeredgecolor="#0d4f0d", markeredgewidth=0.5)


def draw_rectangular(ax):
    track = RectangularTrack(width=10.0, height=8.0, track_width=2.0)
    _draw_background(ax, track)
    _place_car(ax, 2.5, -2.0, -0.3)
    _mark_goal(ax, *track.goal_position)
    ax.set_title("Rectangular Track", fontsize=12, fontweight="bold")


def draw_circular(ax):
    track = CircularTrack(radius=6.0, track_width=2.0)
    _draw_background(ax, track)
    _place_car(ax, 3.0, -3.5, 0.8)
    _mark_goal(ax, *track.goal_position)
    ax.set_title("Circular Track", fontsize=12, fontweight="bold")


def draw_figure8(ax):
    track = Figure8Track(radius=6.0, track_width=2.0)
    _draw_background(ax, track)
    _place_car(ax, 0.0, 3.0, -0.5)
    _mark_goal(ax, *track.goal_position)
    ax.set_title("Figure-8 Track", fontsize=12, fontweight="bold")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    draw_rectangular(axes[0])
    draw_circular(axes[1])
    draw_figure8(axes[2])

    for ax in axes:
        ax.set_xlabel("X position", fontsize=10)
        ax.set_ylabel("Y position", fontsize=10)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="Car (position)"),
        Line2D([0], [0], color="red", linewidth=2,
               label="Car heading (direction)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1a7c1a",
               markersize=12, label="Start / Finish line"),
        Patch(facecolor="#dddddd", edgecolor="#888888", label="Road surface"),
        Line2D([0], [0], color="#666666", linewidth=0.8, label="Road boundaries"),
        Line2D([0], [0], linestyle="--", color="#aaaaaa", linewidth=0.5,
               label="Centerline"),
    ]

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=6, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("NumPy RL Racer — Environment Overview",
                 fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig("images/environment_overview.png", dpi=150,
                bbox_inches="tight", pad_inches=0.3)
    print("Saved images/environment_overview.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
