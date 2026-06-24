"""Record a side-by-side GIF comparing the trained DQN policy vs a random policy."""

import argparse
import os

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import CircularTrack, RacingEnv
from numpy_rl_racer.rendering import MatplotlibRenderer


def _record_episode(env, get_action, track, max_steps):
    renderer = MatplotlibRenderer(track, headless=True)
    renderer.start_recording()
    state = env.reset(seed=42)
    for step in range(max_steps):
        action_idx = get_action(state)
        next_state, reward, done, _ = env.step(ACTIONS[action_idx])
        renderer.render(env.state, step=step, reward=reward)
        state = next_state
        if done:
            break
    renderer.render(env.state, step=step, reward=reward)
    frames = renderer._recording_frames.copy()
    renderer.close()
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare trained vs random policy with a side-by-side GIF"
    )
    parser.add_argument("--model-path", default="models/best_model.npz")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-dir", default="images")
    parser.add_argument("--track", choices=["rectangular", "circular"], default="rectangular")
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.track == "circular":
        track = CircularTrack(radius=6.0, track_width=2.0)
        env = RacingEnv(track=track)
    else:
        track = env = RacingEnv().track
        env = RacingEnv()

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
        state_dim = data["layer_0_w"].shape[0]
        agent = DQNAgent(state_dim=state_dim)
    agent.load(args.model_path)
    agent.epsilon = 0.0

    rng = np.random.RandomState(0)

    trained_frames = _record_episode(
        env, lambda s: agent.act(s, training=False), track, args.max_steps
    )
    random_frames = _record_episode(
        env, lambda s: rng.randint(len(ACTIONS)), track, args.max_steps
    )

    from PIL import Image, ImageDraw

    n_frames = min(len(trained_frames), len(random_frames))
    side_by_side = []
    for i in range(n_frames):
        h_t, w_t = trained_frames[i].shape[:2]
        h_r, w_r = random_frames[i].shape[:2]
        h = max(h_t, h_r)
        w_total = w_t + w_r
        canvas = np.zeros((h, w_total, 3), dtype=np.uint8)
        canvas[:h_t, :w_t] = trained_frames[i]
        canvas[:h_r, w_t:w_total] = random_frames[i]
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.text((8, 6), "Trained policy", fill=(0, 200, 0))
        draw.text((w_t + 8, 6), "Random policy", fill=(200, 0, 0))
        side_by_side.append(img)

    gif_path = os.path.join(args.save_dir, "trained_vs_random.gif")
    duration = 100
    side_by_side[0].save(
        gif_path,
        save_all=True,
        append_images=side_by_side[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
