import argparse
import csv
import math
import os
from dataclasses import dataclass

import numpy as np

from numpy_rl_racer.agent import ACTIONS, DQNAgent
from numpy_rl_racer.env import CircularTrack, RacingEnv

from train import _select_action, main as train_main


@dataclass
class BenchmarkResult:
    best_eval_reward: float
    smoke_eval_reward: float
    smoke_distance: float
    smoke_steps: int
    smoke_passed: bool


@dataclass
class BenchmarkConfig:
    episodes: int = 500
    max_steps: int = 300
    eval_freq: int = 50
    eval_episodes: int = 3
    seed: int = 0
    save_dir: str = "models/v0"
    log_dir: str = "logs/v0"
    batch_size: int = 64
    buffer_size: int = 10000
    epsilon_decay: float = 0.9995
    smoke_steps: int = 120
    smoke_distance_threshold: float = 5.0


def _best_eval_reward(log_dir):
    path = os.path.join(log_dir, "training_log.csv")
    best = -math.inf
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            value = row.get("eval_reward_mean", "")
            if value:
                best = max(best, float(value))
    return best


def _load_agent(model_path, state_dim):
    with np.load(model_path) as data:
        hidden_sizes = list(data["hidden_sizes"]) if "hidden_sizes" in data else [64, 64]
        use_dueling_dqn = bool(int(data["use_dueling_dqn"])) if "use_dueling_dqn" in data else False
        saved_state_dim = int(data["state_dim"]) if "state_dim" in data else state_dim

    agent = DQNAgent(
        state_dim=saved_state_dim,
        hidden_sizes=hidden_sizes,
        use_double_dqn=False,
        use_dueling_dqn=use_dueling_dqn,
    )
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent


def _greedy_eval_distance(model_path, seed, max_steps):
    env = RacingEnv(
        track=CircularTrack(radius=6.0, track_width=2.0),
        randomize_start=False,
        observation_mode="local",
        reward_mode="progress",
        num_reward_lines=0,
    )
    agent = _load_agent(model_path, env.observation_dim)

    state = env.reset(seed=seed, randomize_start=False)
    progress = env.current_progress
    total_reward = 0.0
    forward_distance = 0.0
    steps = 0

    for step in range(max_steps):
        action_idx = _select_action(
            agent,
            state,
            training=False,
            allow_idle_actions=False,
        )
        state, reward, done, _ = env.step(ACTIONS[action_idx])

        progress_delta = env.current_progress - progress
        progress = env.current_progress
        if progress_delta < -0.5:
            progress_delta += 1.0
        if progress_delta > 0.0:
            forward_distance += progress_delta * env.track._perimeter

        total_reward += reward
        steps = step + 1
        if done:
            break

    return float(total_reward), float(forward_distance), steps


def run_benchmark(config):
    train_main([
        "--track", "circular",
        "--reward-mode", "progress",
        "--observation-mode", "local",
        "--num-reward-lines", "0",
        "--no-randomize-start",
        "--episodes", str(config.episodes),
        "--max-steps", str(config.max_steps),
        "--eval-freq", str(config.eval_freq),
        "--eval-episodes", str(config.eval_episodes),
        "--save-dir", config.save_dir,
        "--log-dir", config.log_dir,
        "--seed", str(config.seed),
        "--batch-size", str(config.batch_size),
        "--buffer-size", str(config.buffer_size),
        "--epsilon-decay", str(config.epsilon_decay),
    ])

    best_model_path = os.path.join(config.save_dir, "best_model.npz")
    best_eval_reward = _best_eval_reward(config.log_dir)
    smoke_reward, smoke_distance, smoke_steps = _greedy_eval_distance(
        best_model_path,
        seed=config.seed + 10_000,
        max_steps=config.smoke_steps,
    )
    smoke_passed = bool(smoke_distance > config.smoke_distance_threshold)

    return BenchmarkResult(
        best_eval_reward=best_eval_reward,
        smoke_eval_reward=smoke_reward,
        smoke_distance=smoke_distance,
        smoke_steps=smoke_steps,
        smoke_passed=smoke_passed,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the reproducible circular no-idle v0 baseline benchmark."
    )
    parser.add_argument("--episodes", type=int, default=BenchmarkConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=BenchmarkConfig.max_steps)
    parser.add_argument("--eval-freq", type=int, default=BenchmarkConfig.eval_freq)
    parser.add_argument("--eval-episodes", type=int, default=BenchmarkConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    parser.add_argument("--save-dir", default=BenchmarkConfig.save_dir)
    parser.add_argument("--log-dir", default=BenchmarkConfig.log_dir)
    parser.add_argument("--batch-size", type=int, default=BenchmarkConfig.batch_size)
    parser.add_argument("--buffer-size", type=int, default=BenchmarkConfig.buffer_size)
    parser.add_argument("--epsilon-decay", type=float, default=BenchmarkConfig.epsilon_decay)
    parser.add_argument("--smoke-steps", type=int, default=BenchmarkConfig.smoke_steps)
    parser.add_argument(
        "--smoke-distance-threshold",
        type=float,
        default=BenchmarkConfig.smoke_distance_threshold,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    result = run_benchmark(BenchmarkConfig(**vars(args)))

    print("\nv0 benchmark summary")
    print(f"  best_eval_reward: {result.best_eval_reward:.2f}")
    print(f"  smoke_eval_reward: {result.smoke_eval_reward:.2f}")
    print(f"  smoke_distance: {result.smoke_distance:.2f}")
    print(f"  smoke_steps: {result.smoke_steps}")
    print(f"  smoke_threshold: {args.smoke_distance_threshold:.2f}")
    print(f"  smoke_passed: {'yes' if result.smoke_passed else 'no'}")

    if not result.smoke_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
