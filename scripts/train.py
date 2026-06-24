import argparse
import json
import os

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import CircularTrack, RacingEnv
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
    parser.add_argument("--track", choices=["rectangular", "circular"], default="rectangular",
                        help="Track type to use (default: rectangular)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer sizes")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Replay buffer capacity")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Epsilon decay rate per step")
    parser.add_argument("--target-update-freq", type=int, default=100,
                        help="Target network update frequency (steps)")
    parser.add_argument("--no-double-dqn", action="store_true",
                        help="Disable Double DQN (enabled by default)")
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
        env = RacingEnv(track=track)
    else:
        env = RacingEnv()

    print(f"Track type: {args.track}")
    scheduler = None
    if args.lr_scheduler == "exponential":
        scheduler = ExponentialDecay(args.lr, args.lr_decay)
    elif args.lr_scheduler == "step":
        scheduler = StepDecay(args.lr, args.lr_decay, args.lr_drop_every)
    agent = DQNAgent(
        state_dim=6,
        hidden_sizes=args.hidden_sizes,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        use_double_dqn=not args.no_double_dqn,
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
        f"double_dqn={not args.no_double_dqn}, use_per={args.use_per}, "
        f"dueling_dqn={args.dueling_dqn}, n_step={args.n_step}, "
        f"lr_scheduler={scheduler_str}"
    )

    logger = None
    if args.log_dir:
        from numpy_rl_racer.utils.logging import TrainingLogger
        fieldnames = ["episode", "total_reward", "steps", "avg_loss", "epsilon", "avg_q_value"]
        if args.eval_freq > 0:
            fieldnames.extend(["eval_reward_mean", "eval_reward_std"])
        if args.lr_scheduler != "none":
            fieldnames.append("lr")
        logger = TrainingLogger(os.path.join(args.log_dir, "training_log.csv"),
                                fieldnames=fieldnames)

    episode_rewards = []
    episode_losses = []
    best_reward = -float("inf")
    eval_at_episodes = []
    eval_reward_means = []
    eval_reward_stds = []

    for ep in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed)
        if args.seed is not None:
            args.seed += 1

        ep_reward = 0.0
        ep_losses = []
        ep_q_vals = []

        for step in range(args.max_steps):
            action_idx = agent.act(state)
            next_state, reward, done, _ = env.step(ACTIONS[action_idx])
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
        )
        if args.lr_scheduler != "none":
            log_kwargs["lr"] = agent.optimizer.lr

        if args.eval_freq > 0 and ep % args.eval_freq == 0:
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            _eval_rewards = []
            for _ in range(args.eval_episodes):
                state = env.reset(seed=args.seed)
                if args.seed is not None:
                    args.seed += 1
                _ep_eval_reward = 0.0
                for _ in range(args.max_steps):
                    action_idx = agent.act(state)
                    next_state, reward, done, _ = env.step(ACTIONS[action_idx])
                    _ep_eval_reward += reward
                    state = next_state
                    if done:
                        break
                _eval_rewards.append(_ep_eval_reward)
            agent.epsilon = original_epsilon
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

    if logger:
        logger.close()

    agent.save(os.path.join(args.save_dir, "final_model.npz"))
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print(f"Models saved to {args.save_dir}/")

    plot_training(episode_rewards, episode_losses, args.save_dir,
                  eval_at_episodes=eval_at_episodes,
                  eval_reward_means=eval_reward_means,
                  eval_reward_stds=eval_reward_stds)


if __name__ == "__main__":
    main()
