"""Render a checkpoint-evolution video showing the policy improving over training."""

import argparse
import glob
import os
import re

import numpy as np

from numpy_rl_racer.agent import ACTIONS, DQNAgent
from numpy_rl_racer.rendering import MatplotlibRenderer

# Reuse the same env-building helpers as evaluate.py so checkpoints are evaluated
# with the same track/observation/reward configuration recorded during training.
from evaluate import _build_racing_env, _load_config, _make_track


_EPISODE_RE = re.compile(r"ep(\d+)")


def _parse_episode(path):
    """Extract the embedded `ep{N}` episode number from a checkpoint filename."""
    match = _EPISODE_RE.search(os.path.basename(path))
    if match is None:
        raise ValueError(
            f"Could not parse episode number from checkpoint filename {path!r}; "
            "expected an `ep{{N}}` suffix (e.g. checkpoint_ep5.npz)."
        )
    return int(match.group(1))


def _gather_checkpoints(args):
    if args.checkpoints:
        return list(args.checkpoints)

    pattern = os.path.join(args.checkpoint_dir, args.checkpoint_pattern)
    matches = glob.glob(pattern)
    if not matches:
        raise ValueError(
            f"No checkpoint files found matching pattern {pattern!r}. "
            "Train with --checkpoint-freq N>0 to save periodic snapshots, or "
            "pass explicit --checkpoints paths."
        )
    return sorted(matches, key=_parse_episode)


def _load_agent(model_path):
    """Build a DQNAgent from saved architecture metadata and load its weights."""
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


def _record_episode(renderer, env, agent, max_steps, seed):
    """Run a greedy rollout, recording frames on `renderer`. Returns (frames, reward)."""
    state = env.reset(seed=seed)
    total_reward = 0.0
    reward = 0.0
    info = {"lap_count": env.lap_count, "reward_lines_crossed": 0}
    step = 0
    for step in range(max_steps):
        action_idx = agent.act(state, training=False)
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
    renderer._recording_frames = []
    return frames, total_reward


def _label_frames(frames, episode):
    """Stamp each frame with the source checkpoint episode number."""
    from PIL import Image, ImageDraw

    labeled = []
    for frame in frames:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((8, 6), f"Episode {episode}", fill=(0, 200, 0))
        labeled.append(img)
    return labeled


def _save_gif(images, path, fps):
    if not images:
        raise ValueError("No frames to write; checkpoint rollouts produced empty images.")
    duration = int(1000 / fps)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render a checkpoint-evolution video showing the policy improving over training."
    )
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Explicit checkpoint .npz paths to render, in the order given.",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory to glob for checkpoint snapshots written by --checkpoint-freq.",
    )
    parser.add_argument(
        "--checkpoint-pattern", default="checkpoint_ep*.npz",
        help="Glob pattern used with --checkpoint-dir (default: checkpoint_ep*.npz).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to training config JSON. Defaults to config.json next to the checkpoints.",
    )
    parser.add_argument(
        "--track-seed", type=int, default=None,
        help="Override procedural track seed from config.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-dir", default="images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--mp4", action="store_true",
                        help="Save an MP4 video instead of a GIF (requires ffmpeg).")
    args = parser.parse_args(argv)

    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")
    if args.record_fps <= 0:
        raise ValueError(f"--record-fps must be > 0, got {args.record_fps}")

    if not args.checkpoints and not args.checkpoint_dir:
        raise ValueError(
            "Provide either --checkpoints <paths...> or --checkpoint-dir <dir>."
        )

    checkpoint_paths = _gather_checkpoints(args)

    # Locate the training config: explicit override, or config.json next to the
    # first checkpoint, falling back to the checkpoint directory itself.
    config_path = args.config
    if config_path is None:
        search_dir = (
            os.path.dirname(checkpoint_paths[0])
            if checkpoint_paths else args.checkpoint_dir
        )
        candidate = os.path.join(search_dir, "config.json")
        config_path = candidate if os.path.exists(candidate) else None
    config = _load_config(config_path) if config_path else {}

    if args.track_seed is not None:
        config["track_seed"] = args.track_seed

    track = _make_track(config)
    env = _build_racing_env(config, track)

    # One shared headless renderer accumulates frames from every checkpoint so
    # the MP4 path can call save_video exactly once on a single renderer.
    renderer = MatplotlibRenderer(
        track,
        headless=True,
        reward_line_progress=getattr(env, "_reward_line_progress", None),
    )
    renderer.start_recording()

    all_frames = []
    rewards = []
    episode_labels = []

    for path in checkpoint_paths:
        episode = _parse_episode(path)
        agent = _load_agent(path)
        if env.observation_dim != agent.state_dim:
            raise ValueError(
                f"Checkpoint {path} expects state_dim={agent.state_dim}, but the eval "
                f"env produces observation_dim={env.observation_dim}. Use the training "
                f"config that matches these checkpoints."
            )
        frames, total_reward = _record_episode(
            renderer, env, agent, args.max_steps, args.seed
        )
        all_images = _label_frames(frames, episode)
        all_frames.extend(all_images)
        rewards.append(total_reward)
        episode_labels.append(episode)
        print(f"ep={episode:5d}  reward={total_reward:8.2f}  frames={len(frames):3d}")

    renderer.close()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mp4:
        output_path = os.path.join(args.save_dir, "checkpoint_evolution.mp4")
        # Re-mount the labeled frames onto the renderer for MP4 export.
        renderer._recording_frames = [np.asarray(img, dtype=np.uint8) for img in all_frames]
        renderer.save_video(output_path, fps=args.record_fps)
    else:
        output_path = os.path.join(args.save_dir, "checkpoint_evolution.gif")
        _save_gif(all_frames, output_path, fps=args.fps)

    print(f"\nRendered {len(checkpoint_paths)} checkpoints")
    print(f"Output: {output_path}")
    print("Per-checkpoint mean rollout reward:")
    for ep, reward in zip(episode_labels, rewards):
        print(f"  ep={ep:5d}  reward={reward:8.2f}")


if __name__ == "__main__":
    main()