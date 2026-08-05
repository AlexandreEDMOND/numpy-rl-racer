import argparse
import csv
import json
import os
import time

import numpy as np

from numpy_rl_racer.agent import ACTIONS, DQNAgent
from numpy_rl_racer.env import Obstacle, ProceduralTrack, RacingEnv
from numpy_rl_racer.rendering import MatplotlibRenderer


ACCELERATING_ACTIONS = {0, 1, 2}

SUMMARY_COLUMNS = [
    "seed",
    "episodes",
    "mean_reward",
    "std_reward",
    "mean_steps",
    "std_steps",
    "laps_completed_total",
]


def _infer_state_dim(path):
    data = np.load(path)
    return data["layer_0_w"].shape[0]


def _load_config(path):
    with open(path) as f:
        return json.load(f)


def _generate_obstacles(track, num_obstacles, seed=None):
    rng = np.random.RandomState(seed)
    obstacles = []
    for _ in range(num_obstacles):
        cx, cy, tangent = track.sample_centerline_point(rng=rng)
        lateral = rng.uniform(-0.25, 0.25) * float(track.track_width)
        perp_angle = tangent + np.pi / 2.0
        obstacles.append(Obstacle(
            cx + lateral * np.cos(perp_angle),
            cy + lateral * np.sin(perp_angle),
            rng.uniform(0.3, 0.5),
        ))
    return obstacles


def _make_track(config):
    return ProceduralTrack(
        seed=config.get("track_seed", 0),
        radius=config.get("track_radius", 6.0),
        track_width=2.0,
        num_control_points=config.get("track_points", 12),
        radial_noise=config.get("track_variation", 0.28),
        smoothing_steps=config.get("track_smoothing", 3),
    )


def _build_racing_env(config, track):
    num_obstacles = int(config.get("num_obstacles", 0))
    obstacles = None
    if num_obstacles > 0:
        obstacles = _generate_obstacles(track, num_obstacles, config.get("obstacle_seed"))

    kwargs = dict(
        track=track,
        randomize_start=config.get("randomize_start", True),
        time_penalty=config.get("time_penalty", 0.0),
        obstacles=obstacles,
        num_reward_lines=config.get("num_reward_lines", 0),
        observation_mode=config.get("observation_mode", "state"),
        reward_mode=config.get("reward_mode", "legacy"),
        progress_reward_scale=config.get("progress_reward_scale", 10.0),
        lap_bonus=config.get("lap_bonus", 5.0),
        off_track_penalty=config.get("off_track_penalty", 5.0),
        collision_penalty=config.get("collision_penalty", 5.0),
        step_penalty=config.get("step_penalty", 0.0),
    )
    local_ray_angles = config.get("local_ray_angles")
    if local_ray_angles is not None:
        kwargs["local_ray_angles"] = local_ray_angles
    lidar_max_range = config.get("lidar_max_range")
    if lidar_max_range is not None:
        kwargs["lidar_max_range"] = lidar_max_range
    return RacingEnv(**kwargs)


def _make_env(args, config):
    track_name = args.track if args.track is not None else config.get("track", "procedural")
    if track_name != "procedural":
        raise ValueError(f"Unsupported track {track_name!r}; only 'procedural' is available.")
    config = dict(config)
    if args.track_seed is not None:
        config["track_seed"] = args.track_seed
    track = _make_track(config)
    env = _build_racing_env(config, track)
    return env, track_name


def _select_action(agent, state, allow_idle_actions=True):
    if allow_idle_actions:
        return agent.act(state, training=False)

    allowed = np.array(sorted(ACCELERATING_ACTIONS), dtype=np.int64)
    q_values = agent.online_net.forward(state.reshape(1, -1)).flatten()
    return int(allowed[np.argmax(q_values[allowed])])


def _load_agent(args):
    data = np.load(args.model_path)
    if "arch_type" in data:
        arch_type = int(data["arch_type"])
        hidden_sizes = list(data["hidden_sizes"])
        state_dim = int(data["state_dim"])
        agent = DQNAgent(
            state_dim=state_dim,
            hidden_sizes=hidden_sizes,
            use_dueling_dqn=(arch_type == 1),
        )
    else:
        print("[WARNING] No architecture metadata in checkpoint — assuming MLP architecture.")
        state_dim = data["layer_0_w"].shape[0]
        agent = DQNAgent(state_dim=state_dim)
    agent.load(args.model_path)
    agent.epsilon = 0.0
    return agent


def _run_episodes(env, agent, args, renderer, allow_idle_actions, n_episodes,
                  seed_base, file_prefix):
    total_rewards = []
    total_steps = []
    total_laps = 0

    for ep in range(1, n_episodes + 1):
        state = env.reset(seed=seed_base + ep)
        ep_reward = 0.0

        if args.gif or args.mp4:
            renderer.start_recording()

        reward = 0.0
        info = {"lap_count": env.lap_count, "reward_lines_crossed": 0}
        for step in range(args.max_steps):
            action_idx = _select_action(agent, state, allow_idle_actions=allow_idle_actions)
            next_state, reward, done, info = env.step(ACTIONS[action_idx])
            ep_reward += reward
            renderer.render(
                env.state,
                step=step,
                reward=reward,
                total_reward=ep_reward,
                obstacles=env.obstacles,
                lap_count=info.get("lap_count"),
                reward_lines_crossed=info.get("reward_lines_crossed"),
            )
            if args.live and args.fps > 0:
                time.sleep(1.0 / args.fps)
            state = next_state
            if done:
                break

        total_rewards.append(ep_reward)
        total_steps.append(step + 1)
        total_laps += int(info.get("lap_count", 0))
        print(f"ep={ep:2d}  reward={ep_reward:7.2f}  steps={step + 1:3d}")

        renderer.render(
            env.state,
            step=step,
            reward=reward,
            total_reward=ep_reward,
            obstacles=env.obstacles,
            lap_count=info.get("lap_count"),
            reward_lines_crossed=info.get("reward_lines_crossed"),
        )
        fig_path = os.path.join(args.save_dir, f"{file_prefix}eval_ep{ep}_final.png")
        renderer.fig.savefig(fig_path, dpi=150)
        print(f"  Saved {fig_path}")

        if args.gif:
            gif_path = os.path.join(args.save_dir, f"{file_prefix}eval_ep{ep}.gif")
            renderer.save_animation(gif_path, fps=args.record_fps)
            print(f"  Saved {gif_path}")

        if args.mp4:
            mp4_path = os.path.join(args.save_dir, f"{file_prefix}eval_ep{ep}.mp4")
            renderer.save_video(mp4_path, fps=args.record_fps)
            print(f"  Saved {mp4_path}")

        if args.gif or args.mp4:
            renderer.stop_recording()

    return total_rewards, total_steps, total_laps


def _write_summary(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary_table(rows):
    print("\nPer-seed summary:")
    header = f"{'seed':>6} {'episodes':>8} {'mean_reward':>12} {'std_reward':>11} " \
            f"{'mean_steps':>11} {'std_steps':>10} {'laps':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['seed']:>6} {row['episodes']:>8} "
            f"{row['mean_reward']:>12.2f} {row['std_reward']:>11.2f} "
            f"{row['mean_steps']:>11.1f} {row['std_steps']:>10.1f} "
            f"{row['laps_completed_total']:>6}"
        )


def _plot_multi_seed_summary(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [int(r["seed"]) for r in rows]
    mean_rewards = [float(r["mean_reward"]) for r in rows]
    std_rewards = [float(r["std_reward"]) for r in rows]
    laps = [int(r["laps_completed_total"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seeds))

    ax1.bar(x, mean_rewards, yerr=std_rewards, capsize=4, color="steelblue",
            alpha=0.85, label="mean reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in seeds])
    ax1.set_xlabel("Track seed")
    ax1.set_ylabel("Mean reward (+/- std)")
    ax1.set_title("Generalization across held-out track seeds")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, laps, "o-", color="darkorange", linewidth=2, markersize=6,
             label="laps completed")
    ax2.set_ylabel("Laps completed (total)")
    ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved summary plot to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent in the RacingEnv.")
    parser.add_argument("--model-path", default="models/best_model.npz", help="Path to saved model parameters")
    parser.add_argument("--config", default=None,
                        help="Path to training config JSON. Defaults to config.json next to the model.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--save-dir", default="images", help="Directory to save rendered images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--track", choices=["procedural"], default=None,
                        help="Override track type from config")
    parser.add_argument("--track-seed", type=int, default=None,
                        help="Override procedural track seed from config")
    parser.add_argument("--track-seeds", type=int, nargs="+", default=None,
                        help="Evaluate across multiple held-out track seeds "
                             "(overrides --track-seed when provided)")
    parser.add_argument("--summary", default=None,
                        help="Path to per-seed summary CSV. "
                             "Defaults to <save-dir>/eval_summary.csv.")
    parser.add_argument("--summary-plot", action="store_true", default=False,
                        help="Write a generalization summary plot "
                             "(<save-dir>/eval_summary.png by default) for "
                             "multi-seed evaluation.")
    parser.add_argument("--summary-plot-path", default=None,
                        help="Override path for the multi-seed summary plot. "
                             "Defaults to <save-dir>/eval_summary.png.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI window)")
    parser.add_argument("--live", action="store_true",
                        help="Show the rollout in a live Matplotlib window")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS in live mode")
    parser.add_argument("--gif", "--save-gif", action="store_true",
                        help="Record and save GIF animation of each evaluation episode")
    parser.add_argument("--mp4", "--save-mp4", action="store_true",
                        help="Record and save MP4 video of each evaluation episode")
    parser.add_argument("--record-fps", type=int, default=30,
                        help="FPS for saved GIF/MP4 recordings")
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    config_path = args.config
    if config_path is None:
        candidate = os.path.join(os.path.dirname(args.model_path), "config.json")
        config_path = candidate if os.path.exists(candidate) else None
    config = _load_config(config_path) if config_path else {}

    if args.live:
        args.headless = False

    allow_idle_actions = config.get("allow_idle_actions", True)

    agent = _load_agent(args)
    agent.epsilon = 0.0

    multi_seed = args.track_seeds is not None
    summary_path = args.summary or os.path.join(args.save_dir, "eval_summary.csv")

    if not multi_seed:
        env, track_name = _make_env(args, config)
        print(f"Track type: {track_name}")
        print(f"Observation mode: {env.observation_mode}  Reward mode: {env.reward_mode}")
        print(f"Allow idle actions: {allow_idle_actions}")

        if env.observation_dim != agent.state_dim:
            raise ValueError(
                f"Model expects state_dim={agent.state_dim}, but evaluation env produces "
                f"{env.observation_dim}. Use the training config that matches this model."
            )

        renderer = MatplotlibRenderer(
            env.track,
            headless=args.headless,
            reward_line_progress=getattr(env, "_reward_line_progress", None),
        )

        total_rewards, total_steps, total_laps = _run_episodes(
            env, agent, args, renderer, allow_idle_actions,
            args.episodes, args.seed, "",
        )

        if args.live:
            renderer.show()
        renderer.close()

        print(
            f"\nEvaluation over {args.episodes} episodes:\n"
            f"  Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}\n"
            f"  Average steps:  {np.mean(total_steps):.1f} +/- {np.std(total_steps):.1f}"
        )

        seed_label = args.track_seed if args.track_seed is not None else config.get("track_seed", 0)
        _write_summary(summary_path, [{
            "seed": seed_label,
            "episodes": args.episodes,
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_steps": np.mean(total_steps),
            "std_steps": np.std(total_steps),
            "laps_completed_total": total_laps,
        }])
        return

    # Multi-seed (held-out tracks) path.
    print("Track type: procedural (multi-seed)")
    print(f"Observation mode: {config.get('observation_mode', 'state')}  "
          f"Reward mode: {config.get('reward_mode', 'legacy')}")
    print(f"Allow idle actions: {allow_idle_actions}")
    print(f"Evaluating {len(args.track_seeds)} track seeds: {args.track_seeds}")

    summary_rows = []
    all_rewards = []
    all_steps = []

    for seed in args.track_seeds:
        seed_config = dict(config)
        seed_config["track_seed"] = seed
        track = _make_track(seed_config)
        env = _build_racing_env(seed_config, track)

        if env.observation_dim != agent.state_dim:
            raise ValueError(
                f"Model expects state_dim={agent.state_dim}, but track seed {seed} "
                f"produces observation_dim={env.observation_dim}. "
                f"Use the training config that matches this model."
            )

        renderer = MatplotlibRenderer(
            env.track,
            headless=args.headless,
            reward_line_progress=getattr(env, "_reward_line_progress", None),
        )

        print(f"\n=== Track seed {seed} ===")
        total_rewards, total_steps, total_laps = _run_episodes(
            env, agent, args, renderer, allow_idle_actions,
            args.episodes, args.seed, f"seed{seed}_",
        )

        if args.live:
            renderer.show()
        renderer.close()

        all_rewards.extend(total_rewards)
        all_steps.extend(total_steps)
        summary_rows.append({
            "seed": seed,
            "episodes": args.episodes,
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_steps": float(np.mean(total_steps)),
            "std_steps": float(np.std(total_steps)),
            "laps_completed_total": total_laps,
        })

    _write_summary(summary_path, summary_rows)
    _print_summary_table(summary_rows)

    print(
        f"\nEvaluation over {len(args.track_seeds)} seeds "
        f"({len(args.track_seeds) * args.episodes} episodes total):\n"
        f"  Average reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}\n"
        f"  Average steps:  {np.mean(all_steps):.1f} +/- {np.std(all_steps):.1f}"
    )
    print(f"Summary written to {summary_path}")

    if args.summary_plot:
        plot_path = args.summary_plot_path or os.path.join(args.save_dir, "eval_summary.png")
        _plot_multi_seed_summary(summary_rows, plot_path)


if __name__ == "__main__":
    main()