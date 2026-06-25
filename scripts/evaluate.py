import argparse
import os
import sys

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import CircularTrack, RacingEnv
from numpy_rl_racer.rendering import MatplotlibRenderer


def _infer_state_dim(path):
    data = np.load(path)
    return data["layer_0_w"].shape[0]


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
    parser.add_argument("--render-dpi", type=int, default=100,
                        help="DPI for rendered output (default: 100)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for animation output (default: 10)")
    parser.add_argument("--video", action="store_true",
                        help="Record and save MP4 video of each evaluation episode")
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.track == "circular":
        track = CircularTrack(radius=6.0, track_width=2.0)
        env = RacingEnv(track=track)
    else:
        env = RacingEnv()

    print(f"Track type: {args.track}")
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

    renderer = MatplotlibRenderer(env.track, headless=args.headless, dpi=args.render_dpi, fps=args.fps)

    total_rewards = []
    total_steps = []

    for ep in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + ep)
        ep_reward = 0.0

        if args.gif or args.video:
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
            renderer.save_animation(gif_path)
            print(f"  Saved {gif_path}")
        if args.video:
            video_path = os.path.join(args.save_dir, f"eval_ep{ep}.mp4")
            try:
                renderer.save_video(video_path)
                print(f"  Saved {video_path}")
            except (ImportError, RuntimeError) as e:
                print(f"[WARNING] Could not create MP4: {e}. Saving GIF instead.", file=sys.stderr)
                gif_fallback = os.path.join(args.save_dir, f"eval_ep{ep}.gif")
                renderer.save_animation(gif_fallback)
                print(f"  Saved {gif_fallback}")

        if args.gif or args.video:
            renderer.stop_recording()

    renderer.close()

    print(
        f"\nEvaluation over {args.episodes} episodes:\n"
        f"  Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}\n"
        f"  Average steps:  {np.mean(total_steps):.1f} +/- {np.std(total_steps):.1f}"
    )


if __name__ == "__main__":
    main()
