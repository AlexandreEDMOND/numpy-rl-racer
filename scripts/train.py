import argparse
import json
import os
from contextlib import nullcontext

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import Obstacle, ProceduralTrack, RacingEnv, TrackPoolEnv
from numpy_rl_racer.env.wrappers import ActionRepeatEnv
from numpy_rl_racer.utils.scheduler import ExponentialDecay, StepDecay


ACCELERATING_ACTIONS = {0, 1, 2}


def _load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            raise
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object, got {type(config).__name__}")
    return config


def _select_action(agent, state, training=True, allow_idle_actions=True):
    if allow_idle_actions:
        return agent.act(state, training=training)

    allowed = np.array(sorted(ACCELERATING_ACTIONS), dtype=np.int64)
    rng = agent.rng if agent.rng is not None else np.random
    if training and rng.random() < agent.epsilon:
        return int(rng.choice(allowed))

    q_values = agent.online_net.forward(state.reshape(1, -1)).flatten()
    return int(allowed[np.argmax(q_values[allowed])])


def _evaluate_agent(agent, env, episodes, max_steps, seed, allow_idle_actions):
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rewards = []
    for i in range(episodes):
        eval_seed = None if seed is None else seed + i
        state = env.reset(seed=eval_seed)
        total_reward = 0.0
        for _ in range(max_steps):
            action_idx = _select_action(
                agent,
                state,
                training=False,
                allow_idle_actions=allow_idle_actions,
            )
            next_state, reward, done, _ = env.step(ACTIONS[action_idx])
            total_reward += reward
            state = next_state
            if done:
                break
        rewards.append(total_reward)
    agent.epsilon = original_epsilon
    return rewards


def _generate_obstacles(track, num_obstacles, seed=None):
    rng = np.random.RandomState(seed)
    obstacles = []
    for _ in range(num_obstacles):
        cx, cy, tangent = track.sample_centerline_point(rng=rng)
        lateral = rng.uniform(-0.25, 0.25) * float(track.track_width)
        perp_angle = tangent + np.pi / 2.0
        x = cx + lateral * np.cos(perp_angle)
        y = cy + lateral * np.sin(perp_angle)
        obstacles.append(Obstacle(x, y, rng.uniform(0.3, 0.5)))
    return obstacles


def plot_training(episode_rewards, episode_losses, save_dir,
                  eval_at_episodes=None, eval_reward_means=None, eval_reward_stds=None,
                  off_track_rates=None, collision_steps=None,
                  mean_progress=None, laps_completed=None):
    import matplotlib.pyplot as plt

    has_off_track = off_track_rates is not None and len(off_track_rates) > 0
    has_progress = mean_progress is not None and len(mean_progress) > 0
    n_rows = 2
    if has_off_track:
        n_rows += 1
    if has_progress:
        n_rows += 1
    fig_height = 4 * n_rows - 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_height))
    if n_rows == 1:
        axes = [axes]
    axes = list(axes)

    ax1 = axes[0]
    ax2 = axes[1]
    next_idx = 2
    ax3 = None
    ax4 = None
    if has_off_track:
        ax3 = axes[next_idx]
        next_idx += 1
    if has_progress:
        ax4 = axes[next_idx]

    ax1.plot(episode_rewards, alpha=0.4, label="Episode Reward", color="blue")
    if len(episode_rewards) >= 20:
        smoothed = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")
        ax1.plot(np.arange(19, len(episode_rewards)), smoothed, "r-", linewidth=2, label="Moving avg (20 ep)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Training Rewards")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    if eval_reward_means is not None and len(eval_reward_means) > 0:
        ax1_twin = ax1.twinx()
        ax1_twin.errorbar(eval_at_episodes, eval_reward_means, yerr=eval_reward_stds,
                          fmt="s-", color="purple", label="Eval Reward", markersize=4)
        ax1_twin.set_ylabel("Eval Reward")
        ax1_twin.legend(loc="upper right")

    non_zero = [x for x in episode_losses if x > 0]
    if non_zero:
        ax2.plot(episode_losses, alpha=0.4, label="Avg Loss", color="green")
        if len(episode_losses) >= 20:
            smoothed_l = np.convolve(episode_losses, np.ones(20) / 20, mode="valid")
            ax2.plot(np.arange(19, len(episode_losses)), smoothed_l, "orange", linewidth=2, label="Moving avg (20 ep)")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    if ax3 is not None:
        ax3.plot(np.asarray(off_track_rates), alpha=0.4, label="Off-track Rate", color="red")
        if len(off_track_rates) >= 20:
            smoothed_ot = np.convolve(off_track_rates, np.ones(20) / 20, mode="valid")
            ax3.plot(np.arange(19, len(off_track_rates)), smoothed_ot, "m-", linewidth=2, label="Moving avg (20 ep)")
        if collision_steps is not None and len(collision_steps) > 0:
            ax3.plot(np.asarray(collision_steps), alpha=0.6, label="Collision Steps", color="orange")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Off-track Rate / Collision Steps")
        ax3.set_title("Off-track Rate / Collisions")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)

    if ax4 is not None:
        ax4.plot(np.asarray(mean_progress), alpha=0.4,
                 label="Mean Progress", color="teal")
        if len(mean_progress) >= 20:
            smoothed_p = np.convolve(mean_progress, np.ones(20) / 20, mode="valid")
            ax4.plot(np.arange(19, len(mean_progress)), smoothed_p,
                     "g-", linewidth=2, label="Moving avg (20 ep)")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Mean Progress")
        ax4.set_title("Per-episode Mean Progress / Laps")
        ax4.set_ylim(0.0, 1.0)
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)
        if laps_completed is not None and len(laps_completed) > 0:
            ax4_twin = ax4.twinx()
            ax4_twin.step(np.arange(len(laps_completed)),
                          np.asarray(laps_completed), where="mid",
                          color="purple", alpha=0.6, label="Laps")
            ax4_twin.set_ylabel("Laps Completed")
            ax4_twin.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curve to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a DQN agent in the RacingEnv.")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to JSON config file for hyperparameters")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--save-dir", default="models", help="Directory to save model parameters")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--log-dir", default=None, help="Directory to save training log CSV")
    parser.add_argument("--track", choices=["procedural"], default="procedural",
                        help="Track type to use")
    parser.add_argument("--track-seed", type=int, default=0,
                        help="Seed used to generate the procedural track")
    parser.add_argument("--track-seeds", type=int, nargs="+", default=None,
                        help="Train across a pool of procedural track seeds "
                             "(overrides --track-seed when provided)")
    parser.add_argument("--track-pool-mode", choices=["round_robin", "random"],
                        default="round_robin",
                        help="Selection mode for TrackPoolEnv when --track-seeds is "
                             "provided (default: round_robin)")
    parser.add_argument("--track-radius", type=float, default=6.0,
                        help="Base radius for the procedural track")
    parser.add_argument("--track-points", type=int, default=12,
                        help="Number of control points for the procedural track")
    parser.add_argument("--track-variation", type=float, default=0.28,
                        help="Radial variation ratio for procedural control points")
    parser.add_argument("--track-smoothing", type=int, default=3,
                        help="Number of Chaikin smoothing passes for the procedural track")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer sizes")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Replay buffer capacity")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995,
                        help="Epsilon decay rate per step")
    parser.add_argument("--target-update-freq", type=int, default=100,
                        help="Target network update frequency (steps)")
    parser.add_argument("--double-dqn", action="store_true", default=False,
                        help="Enable Double DQN (disabled by default for the v0 baseline)")
    parser.add_argument("--no-double-dqn", action="store_false", dest="double_dqn",
                        help=argparse.SUPPRESS)
    parser.add_argument("--use-per", action="store_true",
                        help="Enable Prioritized Experience Replay")
    parser.add_argument("--dueling-dqn", action="store_true",
                        help="Enable Dueling DQN architecture")
    parser.add_argument("--noisy-net", action="store_true", default=False,
                        help="Enable NoisyNet exploration (replaces output layers with NoisyLinear)")
    parser.add_argument("--n-step", type=int, default=1,
                        help="N-step returns for TD target (default: 1)")
    parser.add_argument("--lr-scheduler", choices=["none", "exponential", "step"],
                        default="none", help="Learning rate scheduler type (default: none)")
    parser.add_argument("--lr-decay", type=float, default=0.99,
                        help="Decay rate for exponential scheduler, drop factor for step scheduler (default: 0.99)")
    parser.add_argument("--lr-drop-every", type=int, default=100,
                        help="Steps between LR drops for step scheduler (default: 100)")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd",
                        help="Optimizer type: sgd (default) or adam")
    parser.add_argument("--adam-beta1", type=float, default=0.9,
                        help="Adam beta1 (exponential decay rate for first moment), default: 0.9")
    parser.add_argument("--adam-beta2", type=float, default=0.999,
                        help="Adam beta2 (exponential decay rate for second moment), default: 0.999")
    parser.add_argument("--adam-eps", type=float, default=1e-8,
                        help="Adam epsilon for numerical stability, default: 1e-8")
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="Run evaluation every N training episodes (0 = disabled)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes per eval run")
    parser.add_argument("--randomize-start", action="store_true", dest="randomize_start", default=False,
                        help="Enable randomized starts. Disabled by default for the v0 baseline.")
    parser.add_argument("--no-randomize-start", action="store_false", dest="randomize_start",
                        help=argparse.SUPPRESS)
    parser.add_argument("--time-penalty", type=float, default=0.0,
                        help="Time penalty per second of elapsed time (default: 0.0)")
    parser.add_argument("--step-penalty", type=float, default=0.0,
                        help="Fixed reward penalty per step for progress reward mode (default: 0.0)")
    parser.add_argument("--reward-mode", choices=["progress", "legacy"], default="progress",
                        help="Reward mode: progress is the v0 baseline, legacy keeps old shaping")
    parser.add_argument("--progress-reward-scale", type=float, default=10.0,
                        help="Scale applied to progress delta in progress reward mode")
    parser.add_argument("--lap-bonus", type=float, default=5.0,
                        help="Bonus for completing a lap in progress reward mode")
    parser.add_argument("--off-track-penalty", type=float, default=5.0,
                        help="Penalty for leaving the track in progress reward mode")
    parser.add_argument("--collision-penalty", type=float, default=5.0,
                        help="Penalty for obstacle collisions in progress reward mode")
    parser.add_argument("--observation-mode", choices=["local", "state"], default="local",
                        help="Observation mode: local uses car-relative ray inputs for the v0 baseline")
    parser.add_argument("--local-ray-angles", type=float, nargs="*", default=None,
                        help="Car-relative ray angles in radians (observation_mode=local). "
                             "Defaults to the RacingEnv default when not supplied.")
    parser.add_argument("--lidar-max-range", type=float, default=None,
                        help="Maximum lidar ray range forwarded to RacingEnv (default: env default)")
    parser.add_argument("--num-reward-lines", type=int, default=0,
                        help="Checkpoint reward lines; default 0 for the v0 baseline")
    parser.add_argument("--num-obstacles", type=int, default=0,
                        help="Number of obstacles to place on the track (default: 0)")
    parser.add_argument("--obstacle-seed", type=int, default=None,
                        help="Seed for reproducible obstacle placement (default: None)")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Number of times to repeat each action (default: 1)")
    parser.add_argument("--allow-idle-actions", action="store_true", default=False,
                        help="Allow coast/brake actions. Default v0 baseline uses accelerating actions only.")
    parser.add_argument("--checkpoint-freq", type=int, default=0,
                        help="Save a checkpoint every N training episodes (0 = disabled, default: 0)")
    parser.add_argument("--loss-type", choices=["mse", "huber"], default="mse",
                        help="TD-error loss type: mse (default) or huber (smooth-L1)")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Delta for Huber/smooth-L1 loss (default: 1.0)")
    parser.add_argument("--max-grad-norm", type=float, default=None,
                        help="Global L2 gradient-norm clipping threshold "
                             "(default: None disables clipping)")

    known_args, _ = parser.parse_known_args(argv)
    if known_args.config:
        config = _load_config(known_args.config)
        parser.set_defaults(**config)

    args = parser.parse_args(argv)

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    resolved = {k: v for k, v in vars(args).items() if k != "config"}
    config_out = os.path.join(args.save_dir, "config.json")
    with open(config_out, "w") as f:
        json.dump(resolved, f, indent=2)
    print(f"Saved configuration to {config_out}")

    track_kwargs = dict(
        radius=args.track_radius,
        track_width=2.0,
        num_control_points=args.track_points,
        radial_noise=args.track_variation,
        smoothing_steps=args.track_smoothing,
    )
    env_kwargs = dict(
        randomize_start=args.randomize_start,
        time_penalty=args.time_penalty,
        num_reward_lines=args.num_reward_lines,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        progress_reward_scale=args.progress_reward_scale,
        lap_bonus=args.lap_bonus,
        off_track_penalty=args.off_track_penalty,
        collision_penalty=args.collision_penalty,
        step_penalty=args.step_penalty,
    )
    if args.local_ray_angles is not None:
        env_kwargs["local_ray_angles"] = args.local_ray_angles
    if args.lidar_max_range is not None:
        env_kwargs["lidar_max_range"] = args.lidar_max_range

    if args.track_seeds is not None:
        # Multi-track pool path overrides single-seed training.
        print(f"Track pool: seeds={list(args.track_seeds)} mode={args.track_pool_mode}")

        obstacles = None
        if args.num_obstacles > 0:
            rep_track = ProceduralTrack(seed=int(args.track_seeds[0]), **track_kwargs)
            obstacles = _generate_obstacles(rep_track, args.num_obstacles, args.obstacle_seed)
            print(f"Generated {len(obstacles)} obstacles (seed={args.obstacle_seed})")

        pool_env_kwargs = dict(env_kwargs)
        pool_env_kwargs["obstacles"] = obstacles
        env = TrackPoolEnv(
            track_seeds=list(args.track_seeds),
            track_kwargs=track_kwargs,
            seed=args.seed,
            mode=args.track_pool_mode,
            **pool_env_kwargs,
        )

        base_dim = env.envs[0].observation_dim
        for i, sub_env in enumerate(env.envs[1:], start=1):
            sub_dim = sub_env.observation_dim
            if sub_dim != base_dim:
                raise ValueError(
                    f"Pooled tracks have inconsistent observation_dim: track seed "
                    f"{args.track_seeds[0]} has observation_dim={base_dim} but track "
                    f"seed {args.track_seeds[i]} has observation_dim={sub_dim}. "
                    f"observation_dim must match across pooled tracks."
                )
    else:
        track = ProceduralTrack(seed=args.track_seed, **track_kwargs)

        obstacles = None
        if args.num_obstacles > 0:
            obstacles = _generate_obstacles(track, args.num_obstacles, args.obstacle_seed)
            print(f"Generated {len(obstacles)} obstacles (seed={args.obstacle_seed})")

        env = RacingEnv(track=track, obstacles=obstacles, **env_kwargs)

    if args.skip_frames > 1:
        env = ActionRepeatEnv(env, skip_frames=args.skip_frames)
        print(f"Action repeat enabled: skip_frames={args.skip_frames}")
    elif args.skip_frames < 1:
        raise ValueError(f"--skip-frames must be >= 1, got {args.skip_frames}")

    print(f"Track type: {args.track}")
    scheduler = None
    if args.lr_scheduler == "exponential":
        scheduler = ExponentialDecay(args.lr, args.lr_decay)
    elif args.lr_scheduler == "step":
        scheduler = StepDecay(args.lr, args.lr_decay, args.lr_drop_every)
    state_dim = env.observation_dim
    agent = DQNAgent(
        state_dim=state_dim,
        hidden_sizes=args.hidden_sizes,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        use_double_dqn=args.double_dqn,
        use_per=args.use_per,
        use_dueling_dqn=args.dueling_dqn,
        use_noisy=args.noisy_net,
        n_step=args.n_step,
        seed=args.seed,
        scheduler=scheduler,
        optimizer=args.optimizer,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        max_grad_norm=args.max_grad_norm,
    )

    scheduler_str = args.lr_scheduler if args.lr_scheduler != "none" else "none"
    print(
        f"Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, gamma={args.gamma}, "
        f"hidden_sizes={args.hidden_sizes}, buffer_size={args.buffer_size}, "
        f"epsilon_start={args.epsilon_start}, epsilon_min={args.epsilon_min}, "
        f"epsilon_decay={args.epsilon_decay}, target_update_freq={args.target_update_freq}, "
        f"double_dqn={args.double_dqn}, use_per={args.use_per}, "
        f"dueling_dqn={args.dueling_dqn}, noisy_net={args.noisy_net}, n_step={args.n_step}, "
        f"lr_scheduler={scheduler_str}, observation_mode={args.observation_mode}, "
            f"reward_mode={args.reward_mode}, state_dim={state_dim}, "
            f"allow_idle_actions={args.allow_idle_actions}, "
            f"optimizer={args.optimizer}, "
            f"adam_beta1={args.adam_beta1}, adam_beta2={args.adam_beta2}, adam_eps={args.adam_eps}, "
            f"loss_type={args.loss_type}, huber_delta={args.huber_delta}, "
            f"max_grad_norm={args.max_grad_norm}"
    )

    _ctx = nullcontext()
    logger = None
    if args.log_dir:
        from numpy_rl_racer.utils.logging import TrainingLogger
        fieldnames = ["episode", "total_reward", "steps", "avg_loss", "epsilon", "avg_q_value", "elapsed_time",
                      "off_track_steps", "collision_steps", "mean_progress", "laps_completed"]
        if args.eval_freq > 0:
            fieldnames.extend(["eval_reward_mean", "eval_reward_std"])
        if args.lr_scheduler != "none":
            fieldnames.append("lr")
        _ctx = TrainingLogger(os.path.join(args.log_dir, "training_log.csv"),
                                fieldnames=fieldnames)
        logger = _ctx

    episode_rewards = []
    episode_losses = []
    episode_off_track_rates = []
    episode_collision_steps = []
    episode_mean_progress = []
    episode_laps_completed = []
    best_reward = -float("inf")
    best_eval_reward = -float("inf")
    eval_at_episodes = []
    eval_reward_means = []
    eval_reward_stds = []

    with _ctx:
        for ep in range(1, args.episodes + 1):
            state = env.reset(seed=args.seed)
            if args.seed is not None:
                args.seed += 1

            ep_reward = 0.0
            ep_losses = []
            ep_q_vals = []
            off_track_steps = 0
            collision_steps = 0
            ep_progresses = []
            ep_lap_count = 0

            for step in range(args.max_steps):
                action_idx = _select_action(
                    agent,
                    state,
                    training=True,
                    allow_idle_actions=args.allow_idle_actions,
                )
                next_state, reward, done, info = env.step(ACTIONS[action_idx])
                loss = agent.train_step(state, action_idx, reward, next_state, done)
                ep_reward += reward
                if info.get("off_track"):
                    off_track_steps += 1
                if info.get("collision"):
                    collision_steps += 1
                if "progress" in info:
                    ep_progresses.append(float(info["progress"]))
                if "lap_count" in info:
                    ep_lap_count = int(info["lap_count"])
                if loss > 0:
                    ep_losses.append(loss)
                    ep_q_vals.append(agent._last_avg_q)
                state = next_state
                if done:
                    break

            steps_taken = step + 1
            off_track_rate = off_track_steps / steps_taken if steps_taken > 0 else 0.0
            mean_progress = float(np.mean(ep_progresses)) if ep_progresses else 0.0
            laps_completed = ep_lap_count
            avg_loss = np.mean(ep_losses) if ep_losses else float("nan")
            avg_q = np.mean(ep_q_vals) if ep_q_vals else float("nan")
            episode_rewards.append(ep_reward)
            episode_losses.append(avg_loss)
            episode_off_track_rates.append(off_track_rate)
            episode_collision_steps.append(collision_steps)
            episode_mean_progress.append(mean_progress)
            episode_laps_completed.append(laps_completed)

            print(
                f"ep={ep:4d}/{args.episodes}  "
                f"reward={ep_reward:7.2f}  "
                f"loss={avg_loss:.6f}  "
                f"eps={agent.epsilon:.3f}  "
                f"steps={steps_taken:3d}"
            )

            log_kwargs = dict(
                episode=ep,
                total_reward=ep_reward,
                steps=steps_taken,
                avg_loss=avg_loss,
                epsilon=agent.epsilon,
                avg_q_value=avg_q,
                elapsed_time=info.get('elapsed_time', 0.0),
                off_track_steps=off_track_steps,
                collision_steps=collision_steps,
                mean_progress=mean_progress,
                laps_completed=laps_completed,
            )
            if args.lr_scheduler != "none":
                log_kwargs["lr"] = agent.optimizer.lr

            if args.eval_freq > 0 and ep % args.eval_freq == 0:
                eval_seed = None if args.seed is None else args.seed
                _eval_rewards = _evaluate_agent(
                    agent,
                    env,
                    args.eval_episodes,
                    args.max_steps,
                    eval_seed,
                    args.allow_idle_actions,
                )
                if args.seed is not None:
                    args.seed += args.eval_episodes
                eval_mean = np.mean(_eval_rewards)
                eval_std = np.std(_eval_rewards)
                eval_at_episodes.append(ep)
                eval_reward_means.append(eval_mean)
                eval_reward_stds.append(eval_std)
                print(f"  eval: reward={eval_mean:.2f} +/- {eval_std:.2f}")
                log_kwargs["eval_reward_mean"] = eval_mean
                log_kwargs["eval_reward_std"] = eval_std
                if eval_mean > best_eval_reward:
                    best_eval_reward = eval_mean
                    agent.save(os.path.join(args.save_dir, "best_model.npz"))

            if logger:
                logger.log(**log_kwargs)

            if args.eval_freq == 0 and ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(os.path.join(args.save_dir, "best_model.npz"))

            if args.checkpoint_freq > 0 and ep % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(
                    args.save_dir, f"checkpoint_ep{ep}.npz"
                )
                agent.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

    agent.save(os.path.join(args.save_dir, "final_model.npz"))
    if args.eval_freq > 0:
        print(f"\nTraining complete. Best eval reward: {best_eval_reward:.2f}")
    else:
        print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print(f"Models saved to {args.save_dir}/")

    plot_training(episode_rewards, episode_losses, args.save_dir,
                  eval_at_episodes=eval_at_episodes,
                  eval_reward_means=eval_reward_means,
                  eval_reward_stds=eval_reward_stds,
                  off_track_rates=episode_off_track_rates,
                  collision_steps=episode_collision_steps,
                  mean_progress=episode_mean_progress,
                  laps_completed=episode_laps_completed)


if __name__ == "__main__":
    main()
