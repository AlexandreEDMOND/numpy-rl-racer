import argparse
import json
import os
from contextlib import nullcontext

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import CircularTrack, Figure8Track, Obstacle, RacingEnv, RectangularTrack
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
    if hasattr(track, 'radius'):
        R = float(track.radius)
        inner_r = R - track.track_width
        for _ in range(num_obstacles):
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0, inner_r * 0.85)
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            obstacles.append(Obstacle(x, y, rng.uniform(0.3, 0.5)))
    else:
        hw = float(track.half_w)
        hh = float(track.half_h)
        inner_hw = hw - track.track_width
        inner_hh = hh - track.track_width
        for _ in range(num_obstacles):
            x = rng.uniform(-inner_hw * 0.85, inner_hw * 0.85)
            y = rng.uniform(-inner_hh * 0.85, inner_hh * 0.85)
            obstacles.append(Obstacle(x, y, rng.uniform(0.3, 0.5)))
    return obstacles


def plot_training(episode_rewards, episode_losses, save_dir,
                  eval_at_episodes=None, eval_reward_means=None, eval_reward_stds=None):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

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
    parser.add_argument("--track", choices=["rectangular", "circular", "figure8"], default="circular",
                        help="Track type to use (default: circular for the v0 baseline)")
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
    parser.add_argument("--n-step", type=int, default=1,
                        help="N-step returns for TD target (default: 1)")
    parser.add_argument("--lr-scheduler", choices=["none", "exponential", "step"],
                        default="none", help="Learning rate scheduler type (default: none)")
    parser.add_argument("--lr-decay", type=float, default=0.99,
                        help="Decay rate for exponential scheduler, drop factor for step scheduler (default: 0.99)")
    parser.add_argument("--lr-drop-every", type=int, default=100,
                        help="Steps between LR drops for step scheduler (default: 100)")
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

    if args.track == "circular":
        track = CircularTrack(radius=6.0, track_width=2.0)
    elif args.track == "figure8":
        track = Figure8Track(radius=6.0, track_width=2.0)
    else:
        track = RectangularTrack(width=10.0, height=8.0, track_width=2.0)

    obstacles = None
    if args.num_obstacles > 0:
        obstacles = _generate_obstacles(track, args.num_obstacles, args.obstacle_seed)
        print(f"Generated {len(obstacles)} obstacles (seed={args.obstacle_seed})")

    env = RacingEnv(
        track=track,
        randomize_start=args.randomize_start,
        time_penalty=args.time_penalty,
        obstacles=obstacles,
        num_reward_lines=args.num_reward_lines,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        progress_reward_scale=args.progress_reward_scale,
        lap_bonus=args.lap_bonus,
        off_track_penalty=args.off_track_penalty,
        collision_penalty=args.collision_penalty,
        step_penalty=args.step_penalty,
    )

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
        n_step=args.n_step,
        seed=args.seed,
        scheduler=scheduler,
    )

    scheduler_str = args.lr_scheduler if args.lr_scheduler != "none" else "none"
    print(
        f"Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, gamma={args.gamma}, "
        f"hidden_sizes={args.hidden_sizes}, buffer_size={args.buffer_size}, "
        f"epsilon_start={args.epsilon_start}, epsilon_min={args.epsilon_min}, "
        f"epsilon_decay={args.epsilon_decay}, target_update_freq={args.target_update_freq}, "
        f"double_dqn={args.double_dqn}, use_per={args.use_per}, "
        f"dueling_dqn={args.dueling_dqn}, n_step={args.n_step}, "
        f"lr_scheduler={scheduler_str}, observation_mode={args.observation_mode}, "
            f"reward_mode={args.reward_mode}, state_dim={state_dim}, "
            f"allow_idle_actions={args.allow_idle_actions}"
    )

    _ctx = nullcontext()
    logger = None
    if args.log_dir:
        from numpy_rl_racer.utils.logging import TrainingLogger
        fieldnames = ["episode", "total_reward", "steps", "avg_loss", "epsilon", "avg_q_value", "elapsed_time"]
        if args.eval_freq > 0:
            fieldnames.extend(["eval_reward_mean", "eval_reward_std"])
        if args.lr_scheduler != "none":
            fieldnames.append("lr")
        _ctx = TrainingLogger(os.path.join(args.log_dir, "training_log.csv"),
                                fieldnames=fieldnames)
        logger = _ctx

    episode_rewards = []
    episode_losses = []
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
                if loss > 0:
                    ep_losses.append(loss)
                    ep_q_vals.append(agent._last_avg_q)
                state = next_state
                if done:
                    break

            avg_loss = np.mean(ep_losses) if ep_losses else float("nan")
            avg_q = np.mean(ep_q_vals) if ep_q_vals else float("nan")
            episode_rewards.append(ep_reward)
            episode_losses.append(avg_loss)

            print(
                f"ep={ep:4d}/{args.episodes}  "
                f"reward={ep_reward:7.2f}  "
                f"loss={avg_loss:.6f}  "
                f"eps={agent.epsilon:.3f}  "
                f"steps={step + 1:3d}"
            )

            log_kwargs = dict(
                episode=ep,
                total_reward=ep_reward,
                steps=step + 1,
                avg_loss=avg_loss,
                epsilon=agent.epsilon,
                avg_q_value=avg_q,
                elapsed_time=info.get('elapsed_time', 0.0),
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

    agent.save(os.path.join(args.save_dir, "final_model.npz"))
    if args.eval_freq > 0:
        print(f"\nTraining complete. Best eval reward: {best_eval_reward:.2f}")
    else:
        print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print(f"Models saved to {args.save_dir}/")

    plot_training(episode_rewards, episode_losses, args.save_dir,
                  eval_at_episodes=eval_at_episodes,
                  eval_reward_means=eval_reward_means,
                  eval_reward_stds=eval_reward_stds)


if __name__ == "__main__":
    main()
