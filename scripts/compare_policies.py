"""Compare a trained DQN policy against a random policy."""

import argparse
import os

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import ProceduralTrack, RacingEnv
from numpy_rl_racer.rendering import MatplotlibRenderer


def _make_env(track_seed):
    track = ProceduralTrack(seed=track_seed, radius=6.0, track_width=2.0)
    return RacingEnv(track=track)


def _load_agent(model_path):
    data = np.load(model_path)
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
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent


def _record_episode(env, get_action, max_steps, seed):
    renderer = MatplotlibRenderer(
        env.track,
        headless=True,
        reward_line_progress=getattr(env, "_reward_line_progress", None),
    )
    renderer.start_recording()
    state = env.reset(seed=seed)
    total_reward = 0.0
    reward = 0.0
    info = {"lap_count": env.lap_count, "reward_lines_crossed": 0}
    for step in range(max_steps):
        action_idx = get_action(state)
        next_state, reward, done, info = env.step(ACTIONS[action_idx])
        total_reward += reward
        renderer.render(
            env.state,
            step=step,
            reward=reward,
            total_reward=total_reward,
            lap_count=info.get("lap_count"),
            reward_lines_crossed=info.get("reward_lines_crossed"),
        )
        state = next_state
        if done:
            break
    renderer.render(
        env.state,
        step=step,
        reward=reward,
        total_reward=total_reward,
        lap_count=info.get("lap_count"),
        reward_lines_crossed=info.get("reward_lines_crossed"),
    )
    frames = renderer._recording_frames.copy()
    renderer.close()
    return frames


def _save_comparison_gif(agent, args):
    os.makedirs(args.save_dir, exist_ok=True)

    trained_env = _make_env(args.track_seed)
    random_env = _make_env(args.track_seed)
    rng = np.random.RandomState(args.random_seed)

    trained_frames = _record_episode(
        trained_env,
        lambda s: agent.act(s, training=False),
        args.max_steps,
        args.seed,
    )
    random_frames = _record_episode(
        random_env,
        lambda s: rng.randint(len(ACTIONS)),
        args.max_steps,
        args.seed,
    )

    from PIL import Image, ImageDraw

    n_frames = max(len(trained_frames), len(random_frames))
    side_by_side = []
    for i in range(n_frames):
        trained_frame = trained_frames[min(i, len(trained_frames) - 1)]
        random_frame = random_frames[min(i, len(random_frames) - 1)]
        h_t, w_t = trained_frame.shape[:2]
        h_r, w_r = random_frame.shape[:2]
        h = max(h_t, h_r)
        w_total = w_t + w_r
        canvas = np.zeros((h, w_total, 3), dtype=np.uint8)
        canvas[:h_t, :w_t] = trained_frame
        canvas[:h_r, w_t:w_total] = random_frame
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.text((8, 6), "Trained policy", fill=(0, 200, 0))
        draw.text((w_t + 8, 6), "Random policy", fill=(200, 0, 0))
        side_by_side.append(img)

    gif_path = os.path.join(args.save_dir, "trained_vs_random.gif")
    duration = int(1000 / args.fps)
    side_by_side[0].save(
        gif_path,
        save_all=True,
        append_images=side_by_side[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved {gif_path}")


def _step_policy(env, state, action_idx, total_reward):
    next_state, reward, done, info = env.step(ACTIONS[action_idx])
    return next_state, done, info, reward, total_reward + reward


def _render_live_comparison(agent, args):
    import matplotlib.pyplot as plt

    trained_env = _make_env(args.track_seed)
    random_env = _make_env(args.track_seed)
    rng = np.random.RandomState(args.random_seed)

    trained_state = trained_env.reset(seed=args.seed)
    random_state = random_env.reset(seed=args.seed)
    trained_done = False
    random_done = False
    trained_total = 0.0
    random_total = 0.0
    trained_reward = 0.0
    random_reward = 0.0
    trained_info = {"lap_count": trained_env.lap_count, "reward_lines_crossed": 0}
    random_info = {"lap_count": random_env.lap_count, "reward_lines_crossed": 0}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    trained_renderer = MatplotlibRenderer(
        trained_env.track,
        reward_line_progress=getattr(trained_env, "_reward_line_progress", None),
        ax=axes[0],
    )
    random_renderer = MatplotlibRenderer(
        random_env.track,
        reward_line_progress=getattr(random_env, "_reward_line_progress", None),
        ax=axes[1],
    )
    fig.canvas.manager.set_window_title("numpy-rl-racer: trained vs random")

    for step in range(args.max_steps):
        if not plt.fignum_exists(fig.number):
            break

        if not trained_done:
            trained_action = agent.act(trained_state, training=False)
            trained_state, trained_done, trained_info, trained_reward, trained_total = _step_policy(
                trained_env, trained_state, trained_action, trained_total
            )
        if not random_done:
            random_action = rng.randint(len(ACTIONS))
            random_state, random_done, random_info, random_reward, random_total = _step_policy(
                random_env, random_state, random_action, random_total
            )

        trained_renderer.render(
            trained_env.state,
            step=step,
            reward=trained_reward,
            total_reward=trained_total,
            lap_count=trained_info.get("lap_count"),
            reward_lines_crossed=trained_info.get("reward_lines_crossed"),
        )
        axes[0].set_title(f"Trained policy\n{axes[0].get_title()}")

        random_renderer.render(
            random_env.state,
            step=step,
            reward=random_reward,
            total_reward=random_total,
            lap_count=random_info.get("lap_count"),
            reward_lines_crossed=random_info.get("reward_lines_crossed"),
        )
        axes[1].set_title(f"Random policy\n{axes[1].get_title()}")

        fig.tight_layout()
        plt.pause(1.0 / args.fps)

        if trained_done and random_done:
            break

    plt.show(block=True)
    trained_renderer.close()
    random_renderer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare trained vs random policy with GIF or live visualization"
    )
    parser.add_argument("--model-path", default="models/best_model.npz")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-dir", default="images")
    parser.add_argument("--track-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--live", action="store_true",
                        help="Open an interactive side-by-side viewer instead of saving a GIF")
    args = parser.parse_args(argv)

    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")

    agent = _load_agent(args.model_path)
    if args.live:
        _render_live_comparison(agent, args)
    else:
        _save_comparison_gif(agent, args)


if __name__ == "__main__":
    main()
