#!/usr/bin/env python3
"""Grid search over hyperparameters for numpy-rl-racer.

Usage:
  python scripts/grid_search.py --lr 1e-3,5e-3 --gamma 0.99,0.95 --episodes 50 --max-steps 100
  python scripts/grid_search.py --grid-config grid_config.json --episodes 50
  python scripts/grid_search.py --config base.json --lr 1e-3,1e-2 --episodes 100
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from train import main as train_main  # noqa: E402


def _parse_value(v):
    v = v.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_grid_cli(unknown_args):
    grid = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            name = arg[2:].replace("-", "_")
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                val = unknown_args[i + 1]
                if "," in val:
                    grid[name] = [_parse_value(v) for v in val.split(",")]
                i += 2
            else:
                i += 1
        else:
            i += 1
    return grid


def _build_train_args(base_config, combo, episodes, max_steps, run_dir, run_seed):
    args = []
    if base_config:
        args.extend(["--config", base_config])
    args.extend(["--episodes", str(episodes)])
    args.extend(["--max-steps", str(max_steps)])
    args.extend(["--save-dir", run_dir])
    args.extend(["--log-dir", run_dir])
    if run_seed is not None:
        args.extend(["--seed", str(run_seed)])
    for k, v in combo.items():
        if isinstance(v, bool):
            if k == "randomize_start":
                if not v:
                    args.append("--no-randomize-start")
            elif k == "no_randomize_start":
                if v:
                    args.append("--no-randomize-start")
            elif v:
                args.append("--" + k.replace("_", "-"))
        else:
            args.extend(["--" + k.replace("_", "-"), str(v)])
    return args


def _extract_results(run_id, combo, run_dir, elapsed):
    log_csv = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(log_csv):
        return {
            "run_id": run_id,
            "params": json.dumps(combo),
            "final_reward": float("nan"),
            "mean_reward": float("nan"),
            "std_reward": float("nan"),
            "final_loss": float("nan"),
            "mean_loss": float("nan"),
            "total_steps": 0,
            "elapsed_time": elapsed,
        }

    with open(log_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rewards = [float(r["total_reward"]) for r in rows]
    losses = []
    for r in rows:
        v = r.get("avg_loss", "")
        if v:
            losses.append(float(v))
    steps = [int(r["steps"]) for r in rows]

    return {
        "run_id": run_id,
        "params": json.dumps(combo),
        "final_reward": rewards[-1] if rewards else float("nan"),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "std_reward": float(np.std(rewards)) if rewards else float("nan"),
        "final_loss": losses[-1] if losses else float("nan"),
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "total_steps": int(np.sum(steps)),
        "elapsed_time": elapsed,
    }


def _save_csv(results, path):
    fieldnames = [
        "run_id", "params", "final_reward", "mean_reward", "std_reward",
        "final_loss", "mean_loss", "total_steps", "elapsed_time",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {path}")


def _plot_results(results, path):
    import matplotlib.pyplot as plt

    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = [
        ", ".join(f"{k}={v}" for k, v in json.loads(r["params"]).items())
        for r in results
    ]
    x = np.arange(n)

    ax = axes[0, 0]
    ax.bar(x, [r["final_reward"] for r in results], color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final Reward")
    ax.set_title("Final Reward per Configuration")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    mean_vals = [r["mean_reward"] for r in results]
    std_vals = [r["std_reward"] for r in results]
    ax.bar(x, mean_vals, yerr=std_vals, color="seagreen", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward per Configuration")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.bar(x, [r["final_loss"] for r in results], color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss per Configuration")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.bar(x, [r["total_steps"] for r in results], color="mediumpurple")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total Steps")
    ax.set_title("Total Steps per Configuration")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved summary plot to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Grid search over hyperparameters for numpy-rl-racer."
    )
    parser.add_argument("--config", "-c", default=None,
                        help="Base config JSON file (same format as train.py --config)")
    parser.add_argument("--grid-config", default=None,
                        help="JSON file defining grid search parameter space")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (incremented per run)")
    parser.add_argument("--output", "-o", default="grid_search_results.csv",
                        help="Output CSV path")
    parser.add_argument("--plot", default="grid_search_results.png",
                        help="Output plot path")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes per run")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode")

    args, unknown = parser.parse_known_args(argv)

    grid_params = {}
    if args.grid_config:
        with open(args.grid_config) as f:
            grid_params.update(json.load(f))
    grid_params.update(_parse_grid_cli(unknown))

    if not grid_params:
        parser.error(
            "No grid parameters specified. "
            "Use --param-name val1,val2,val3 or --grid-config."
        )

    keys = list(grid_params.keys())
    value_lists = [grid_params[k] for k in keys]

    total = 1
    for vl in value_lists:
        total *= len(vl)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Grid search: {total} combination(s)")
    results = []

    for i, values in enumerate(itertools.product(*value_lists)):
        combo = dict(zip(keys, values))
        print(f"  Run {i + 1}/{total}: {combo}")

        run_dir = os.path.join(output_dir, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)

        run_seed = args.seed + i if args.seed is not None else None

        train_args = _build_train_args(
            args.config, combo, args.episodes, args.max_steps,
            run_dir, run_seed,
        )

        start = time.time()
        log_path = os.path.join(run_dir, "run.log")
        with open(log_path, "w") as log_f:
            with redirect_stdout(log_f), redirect_stderr(log_f):
                try:
                    train_main(train_args)
                except SystemExit:
                    pass
        elapsed = time.time() - start

        result = _extract_results(i, combo, run_dir, elapsed)
        results.append(result)
        print(f"    final_reward={result['final_reward']:.2f}, "
              f"elapsed={result['elapsed_time']:.1f}s")

    _save_csv(results, args.output)
    _plot_results(results, args.plot)


if __name__ == "__main__":
    main()
