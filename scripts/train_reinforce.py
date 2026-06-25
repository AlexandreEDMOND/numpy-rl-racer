import argparse
import json
import os
from contextlib import nullcontext

import numpy as np

from numpy_rl_racer.agent import ACTIONS
from numpy_rl_racer.agent.reinforce import REINFORCEAgent
from numpy_rl_racer.env import CircularTrack, Figure8Track, Obstacle, RacingEnv, RectangularTrack
from numpy_rl_racer.env.wrappers import ActionRepeatEnv
from numpy_rl_racer.utils.scheduler import ExponentialDecay, StepDecay


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
    parser = argparse.ArgumentParser(
        description="Train a REINFORCE (Monte Carlo policy gradient) agent in the RacingEnv."
    )
    parser.add_argument("--config", "-c", default=None,
                        help="Path to JSON config file for hyperparameters")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--save-dir", default="models", help="Directory to save model parameters")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--log-dir", default=None, help="Directory to save training log CSV")
    parser.add_argument("--track", choices=["rectangular", "circular", "figure8"], default="rectangular",
                        help="Track type to use (default: rectangular)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer sizes for policy network")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd",
                        help="Optimizer type (default: sgd)")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="SGD momentum coefficient (default: 0.0)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay (L2 regularization) coefficient (default: 0.0)")
    parser.add_argument("--max-grad-norm", type=float, default=None,
                        help="Maximum global gradient norm for clipping (default: no clipping)")
    parser.add_argument("--no-baseline", action="store_false", dest="use_baseline",
                        help="Disable reward-to-go baseline subtraction (default: enabled)")
    parser.add_argument("--value-network", action="store_true", dest="use_value_network",
                        help="Enable learned baseline via a separate value network (lightweight Actor-Critic)")
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
    parser.add_argument("--no-randomize-start", action="store_false", dest="randomize_start", default=True,
                        help="Disable randomized start position (default: enabled)")
    parser.add_argument("--time-penalty", type=float, default=0.0,
                        help="Time penalty per second of elapsed time (default: 0.0)")
    parser.add_argument("--num-obstacles", type=int, default=0,
                        help="Number of obstacles to place on the track (default: 0)")
    parser.add_argument("--obstacle-seed", type=int, default=None,
                        help="Seed for reproducible obstacle placement (default: None)")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Number of times to repeat each action (default: 1)")

    known_args, _ = parser.parse_known_args(argv)
    if known_args.config:
        config = _load_config(known_args.config)
        parser.set_defaults(**config)

    args = parser.parse_args(argv)

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

    env = RacingEnv(track=track, randomize_start=args.randomize_start,
                    time_penalty=args.time_penalty, obstacles=obstacles)

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
    state_dim = 8 if obstacles else 6
    agent = REINFORCEAgent(
        state_dim=state_dim,
        hidden_sizes=args.hidden_sizes,
        lr=args.lr,
        gamma=args.gamma,
        use_baseline=args.use_baseline,
        use_value_network=args.use_value_network,
        optimizer_type=args.optimizer,
        scheduler=scheduler,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )

    scheduler_str = args.lr_scheduler if args.lr_scheduler != "none" else "none"
    print(
        f"Hyperparameters: lr={args.lr}, gamma={args.gamma}, "
        f"hidden_sizes={args.hidden_sizes}, optimizer={args.optimizer}, "
        f"momentum={args.momentum}, weight_decay={args.weight_decay}, "
        f"max_grad_norm={args.max_grad_norm}, use_baseline={args.use_baseline}, "
        f"use_value_network={args.use_value_network}, "
        f"lr_scheduler={scheduler_str}"
    )

    _ctx = nullcontext()
    logger = None
    if args.log_dir:
        from numpy_rl_racer.utils.logging import TrainingLogger
        fieldnames = ["episode", "total_reward", "steps", "avg_loss", "elapsed_time"]
        if args.use_value_network:
            fieldnames.append("avg_value_loss")
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
    eval_at_episodes = []
    eval_reward_means = []
    eval_reward_stds = []

    with _ctx:
        for ep in range(1, args.episodes + 1):
            state = env.reset(seed=args.seed)
            if args.seed is not None:
                args.seed += 1

            ep_states = []
            ep_actions = []
            ep_rewards = []
            ep_reward = 0.0

            for step in range(args.max_steps):
                action_idx = agent.act(state)
                next_state, reward, done, info = env.step(ACTIONS[action_idx])
                ep_states.append(state)
                ep_actions.append(action_idx)
                ep_rewards.append(reward)
                ep_reward += reward
                state = next_state
                if done:
                    break

            states_arr = np.array(ep_states)
            actions_arr = np.array(ep_actions)
            rewards_arr = np.array(ep_rewards, dtype=np.float64)
            result = agent.train_step(states_arr, actions_arr, rewards_arr)

            if args.use_value_network:
                avg_loss, avg_v_loss = result
            else:
                avg_loss = result
                avg_v_loss = float("nan")

            episode_rewards.append(ep_reward)
            episode_losses.append(avg_loss)

            loss_str = f"loss={avg_loss:.6f}"
            if args.use_value_network:
                loss_str += f"  v_loss={avg_v_loss:.6f}"

            print(
                f"ep={ep:4d}/{args.episodes}  "
                f"reward={ep_reward:7.2f}  "
                f"{loss_str}  "
                f"steps={step + 1:3d}"
            )

            log_kwargs = dict(
                episode=ep,
                total_reward=ep_reward,
                steps=step + 1,
                avg_loss=avg_loss,
                elapsed_time=info.get('elapsed_time', 0.0),
            )
            if args.use_value_network:
                log_kwargs["avg_value_loss"] = avg_v_loss
            if args.lr_scheduler != "none":
                log_kwargs["lr"] = agent.optimizer.lr

            if args.eval_freq > 0 and ep % args.eval_freq == 0:
                _eval_rewards = []
                for _ in range(args.eval_episodes):
                    state = env.reset(seed=args.seed)
                    if args.seed is not None:
                        args.seed += 1
                    _ep_eval_reward = 0.0
                    for _ in range(args.max_steps):
                        action_idx = agent.act(state, training=False)
                        next_state, reward, done, _ = env.step(ACTIONS[action_idx])
                        _ep_eval_reward += reward
                        state = next_state
                        if done:
                            break
                    _eval_rewards.append(_ep_eval_reward)
                eval_mean = np.mean(_eval_rewards)
                eval_std = np.std(_eval_rewards)
                eval_at_episodes.append(ep)
                eval_reward_means.append(eval_mean)
                eval_reward_stds.append(eval_std)
                print(f"  eval: reward={eval_mean:.2f} +/- {eval_std:.2f}")
                log_kwargs["eval_reward_mean"] = eval_mean
                log_kwargs["eval_reward_std"] = eval_std

            if logger:
                logger.log(**log_kwargs)

            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(os.path.join(args.save_dir, "best_model.npz"))

    agent.save(os.path.join(args.save_dir, "final_model.npz"))
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print(f"Models saved to {args.save_dir}/")

    plot_training(episode_rewards, episode_losses, args.save_dir,
                  eval_at_episodes=eval_at_episodes,
                  eval_reward_means=eval_reward_means,
                  eval_reward_stds=eval_reward_stds)


if __name__ == "__main__":
    main()
