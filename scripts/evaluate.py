import argparse
import os

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import CircularTrack, RacingEnv
from numpy_rl_racer.rendering import MatplotlibRenderer


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent in the RacingEnv.")
    parser.add_argument("--model-path", default="models/best_model.npz", help="Path to saved model parameters")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--save-dir", default="images", help="Directory to save rendered images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--track", choices=["rectangular", "circular"], default="rectangular",
                        help="Track type to use (default: rectangular)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI window)")
    parser.add_argument("--gif", "--save-gif", action="store_true",
                        help="Record and save GIF animation of each evaluation episode")
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.track == "circular":
        track = CircularTrack(radius=6.0, track_width=2.0)
        env = RacingEnv(track=track)
    else:
        env = RacingEnv()

    print(f"Track type: {args.track}")
    agent = DQNAgent(state_dim=6)
    agent.load(args.model_path)
    agent.epsilon = 0.0

    renderer = MatplotlibRenderer(env.track, headless=args.headless)

    total_rewards = []
    total_steps = []

    for ep in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + ep)
        ep_reward = 0.0

        if args.gif:
            renderer.start_recording()

        for step in range(args.max_steps):
            action_idx = agent.act(state, training=False)
            next_state, reward, done, _ = env.step(ACTIONS[action_idx])
            renderer.render(env.state, step=step, reward=reward)
            ep_reward += reward
            state = next_state
            if done:
                break

        total_rewards.append(ep_reward)
        total_steps.append(step + 1)
        print(f"ep={ep:2d}  reward={ep_reward:7.2f}  steps={step + 1:3d}")

        renderer.render(env.state, step=step, reward=ep_reward)
        fig_path = os.path.join(args.save_dir, f"eval_ep{ep}_final.png")
        renderer.fig.savefig(fig_path, dpi=150)
        print(f"  Saved {fig_path}")

        if args.gif:
            gif_path = os.path.join(args.save_dir, f"eval_ep{ep}.gif")
            renderer.save_animation(gif_path, fps=10)
            print(f"  Saved {gif_path}")
            renderer.stop_recording()

    renderer.close()

    print(
        f"\nEvaluation over {args.episodes} episodes:\n"
        f"  Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}\n"
        f"  Average steps:  {np.mean(total_steps):.1f} +/- {np.std(total_steps):.1f}"
    )


if __name__ == "__main__":
    main()
