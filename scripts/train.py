import argparse
import os

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import CircularTrack, RacingEnv
from numpy_rl_racer.utils.scheduler import ExponentialDecay, StepDecay


def plot_training(episode_rewards, episode_losses, save_dir):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(episode_rewards, alpha=0.4, label="Episode Reward", color="blue")
    if len(episode_rewards) >= 20:
        smoothed = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")
        ax1.plot(np.arange(19, len(episode_rewards)), smoothed, "r-", linewidth=2, label="Moving avg (20 ep)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Training Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

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
        if args.lr_scheduler != "none":
            fieldnames.append("lr")
        logger = TrainingLogger(os.path.join(args.log_dir, "training_log.csv"),
                                fieldnames=fieldnames)

    episode_rewards = []
    episode_losses = []
    best_reward = -float("inf")

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

        if logger:
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
            logger.log(**log_kwargs)

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(args.save_dir, "best_model.npz"))

    if logger:
        logger.close()

    agent.save(os.path.join(args.save_dir, "final_model.npz"))
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print(f"Models saved to {args.save_dir}/")

    plot_training(episode_rewards, episode_losses, args.save_dir)


if __name__ == "__main__":
    main()
