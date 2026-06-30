"""Render a gallery of procedurally generated tracks."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numpy_rl_racer.env import ProceduralTrack
from numpy_rl_racer.rendering import MatplotlibRenderer


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preview procedural track generation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)),
                        help="Track seeds to render")
    parser.add_argument("--cols", type=int, default=4,
                        help="Number of columns in the output gallery")
    parser.add_argument("--output", default="images/procedural_track_gallery.png",
                        help="Output image path")
    parser.add_argument("--radius", type=float, default=6.0,
                        help="Base track radius")
    parser.add_argument("--track-width", type=float, default=2.0,
                        help="Road width")
    parser.add_argument("--points", type=int, default=12,
                        help="Number of procedural control points")
    parser.add_argument("--variation", type=float, default=0.28,
                        help="Radial variation ratio")
    parser.add_argument("--smoothing", type=int, default=3,
                        help="Number of Chaikin smoothing passes")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.cols <= 0:
        raise ValueError(f"--cols must be > 0, got {args.cols}")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")

    rows = (len(args.seeds) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(4.2 * args.cols, 4.0 * rows))
    axes = getattr(axes, "flat", [axes])

    for ax in axes:
        ax.axis("off")

    for ax, seed in zip(axes, args.seeds):
        ax.axis("on")
        track = ProceduralTrack(
            seed=seed,
            radius=args.radius,
            track_width=args.track_width,
            num_control_points=args.points,
            radial_noise=args.variation,
            smoothing_steps=args.smoothing,
        )
        renderer = MatplotlibRenderer(track, headless=True, ax=ax)
        renderer._draw_background()
        ax.set_title(f"seed={seed}")

    fig.tight_layout()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
