"""Compare a trained DQN policy against a random policy."""

import argparse
import os

import numpy as np

from numpy_rl_racer.agent import ACTIONS, DQNAgent
from numpy_rl_racer.rendering import MatplotlibRenderer

# Reuse the same env-building helpers as evaluate.py so the comparison plays
# back using the same track/observation/reward configuration as training.
from evaluate import (
    ACCELERATING_ACTIONS,
    _build_racing_env,
    _load_config,
    _make_track,
    _select_action,
)


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


def _build_envs(config):
    """Build two envs (trained, random) from the same config/track."""
    track = _make_track(config)
    trained_env = _build_racing_env(config, track)
    random_env = _build_racing_env(config, track)
    return trained_env, random_env


def _random_action_factory(rng, allow_idle_actions):
    if allow_idle_actions:
        n_actions = len(ACTIONS)
        return lambda state: int(rng.randint(n_actions))
    allowed = np.array(sorted(ACCELERATING_ACTIONS), dtype=np.int64)
    return lambda state: int(rng.choice(allowed))


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


def _check_observation_dim(env, agent, label):
    if env.observation_dim != agent.state_dim:
        raise ValueError(
            f"Model expects state_dim={agent.state_dim}, but the {label} env produces "
            f"observation_dim={env.observation_dim}. Use the training config that "
            f"matches this model."
        )


def _save_comparison_gif(agent, args, config, allow_idle_actions):
    os.makedirs(args.save_dir, exist_ok=True)

    trained_env, random_env = _build_envs(config)
    _check_observation_dim(trained_env, agent, "trained-policy")
    _check_observation_dim(random_env, agent, "random-policy")
    rng = np.random.RandomState(args.random_seed)

    def trained_get_action(s):
        return _select_action(agent, s, allow_idle_actions=allow_idle_actions)
    random_get_action = _random_action_factory(rng, allow_idle_actions)

    trained_frames = _record_episode(
        trained_env,
        trained_get_action,
        args.max_steps,
        args.seed,
    )
    random_frames = _record_episode(
        random_env,
        random_get_action,
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


def _render_live_comparison(agent, args, config, allow_idle_actions):
    import matplotlib.pyplot as plt

    trained_env, random_env = _build_envs(config)
    _check_observation_dim(trained_env, agent, "trained-policy")
    _check_observation_dim(random_env, agent, "random-policy")
    rng = np.random.RandomState(args.random_seed)
    random_get_action = _random_action_factory(rng, allow_idle_actions)

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
            trained_action = _select_action(
                agent, trained_state, allow_idle_actions=allow_idle_actions
            )
            trained_state, trained_done, trained_info, trained_reward, trained_total = _step_policy(
                trained_env, trained_state, trained_action, trained_total
            )
        if not random_done:
            random_action = random_get_action(random_state)
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
    parser.add_argument(
        "--config", default=None,
        help="Path to training config JSON. Defaults to config.json next to the model.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-dir", default="images")
    parser.add_argument(
        "--track-seed", type=int, default=None,
        help="Override procedural track seed from config.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--allow-idle-actions", dest="allow_idle_actions", action="store_true",
        default=None,
        help="Restore the full 5-action policy (overrides config). When False the "
             "policies only use the accelerating subset {0, 1, 2}.",
    )
    parser.add_argument(
        "--no-allow-idle-actions", dest="allow_idle_actions", action="store_false",
        help="Restrict to accelerating actions {0, 1, 2} (overrides config).",
    )
    parser.add_argument("--live", action="store_true",
                        help="Open an interactive side-by-side viewer instead of saving a GIF")
    args = parser.parse_args(argv)

    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")

    config_path = args.config
    if config_path is None:
        candidate = os.path.join(os.path.dirname(args.model_path), "config.json")
        config_path = candidate if os.path.exists(candidate) else None
    config = _load_config(config_path) if config_path else {}
    config = dict(config)

    if args.track_seed is not None:
        config["track_seed"] = args.track_seed

    if args.allow_idle_actions is None:
        allow_idle_actions = bool(config.get("allow_idle_actions", False))
    else:
        allow_idle_actions = args.allow_idle_actions

    agent = _load_agent(args.model_path)
    if args.live:
        _render_live_comparison(agent, args, config, allow_idle_actions)
    else:
        _save_comparison_gif(agent, args, config, allow_idle_actions)


if __name__ == "__main__":
    main()