import argparse
import os

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import CircularTrack, RacingEnv


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


def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent in the RacingEnv.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--save-dir", default="models", help="Directory to save model parameters")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--log-dir", default=None, help="Directory to save training log CSV")
    parser.add_argument("--track", choices=["rectangular", "circular"], default="rectangular",
                        help="Track type to use (default: rectangular)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.track == "circular":
        track = CircularTrack(radius=6.0, track_width=2.0)
        env = RacingEnv(track=track)
    else:
        env = RacingEnv()

    print(f"Track type: {args.track}")
    agent = DQNAgent(state_dim=6)

    logger = None
    if args.log_dir:
        from numpy_rl_racer.utils.logging import TrainingLogger
        logger = TrainingLogger(os.path.join(args.log_dir, "training_log.csv"))

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
            logger.log(
                episode=ep,
                total_reward=ep_reward,
                steps=step + 1,
                avg_loss=avg_loss,
                epsilon=agent.epsilon,
                avg_q_value=avg_q,
            )

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
