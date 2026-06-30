import argparse
import json
import os
import time

import numpy as np

from numpy_rl_racer.agent import DQNAgent, ACTIONS
from numpy_rl_racer.env import Obstacle, ProceduralTrack, RacingEnv
from numpy_rl_racer.rendering import MatplotlibRenderer


ACCELERATING_ACTIONS = {0, 1, 2}


def _infer_state_dim(path):
    data = np.load(path)
    return data["layer_0_w"].shape[0]


def _load_config(path):
    with open(path) as f:
        return json.load(f)


def _generate_obstacles(track, num_obstacles, seed=None):
    rng = np.random.RandomState(seed)
    obstacles = []
    for _ in range(num_obstacles):
        cx, cy, tangent = track.sample_centerline_point(rng=rng)
        lateral = rng.uniform(-0.25, 0.25) * float(track.track_width)
        perp_angle = tangent + np.pi / 2.0
        obstacles.append(Obstacle(
            cx + lateral * np.cos(perp_angle),
            cy + lateral * np.sin(perp_angle),
            rng.uniform(0.3, 0.5),
        ))
    return obstacles


def _make_track(config):
    return ProceduralTrack(
        seed=config.get("track_seed", 0),
        radius=config.get("track_radius", 6.0),
        track_width=2.0,
        num_control_points=config.get("track_points", 12),
        radial_noise=config.get("track_variation", 0.28),
        smoothing_steps=config.get("track_smoothing", 3),
    )


def _make_env(args, config):
    track_name = args.track if args.track is not None else config.get("track", "procedural")
    if track_name != "procedural":
        raise ValueError(f"Unsupported track {track_name!r}; only 'procedural' is available.")
    config = dict(config)
    if args.track_seed is not None:
        config["track_seed"] = args.track_seed
    track = _make_track(config)
    num_obstacles = int(config.get("num_obstacles", 0))
    obstacles = None
    if num_obstacles > 0:
        obstacles = _generate_obstacles(track, num_obstacles, config.get("obstacle_seed"))

    env = RacingEnv(
        track=track,
        randomize_start=config.get("randomize_start", True),
        time_penalty=config.get("time_penalty", 0.0),
        obstacles=obstacles,
        num_reward_lines=config.get("num_reward_lines", 0),
        observation_mode=config.get("observation_mode", "state"),
        reward_mode=config.get("reward_mode", "legacy"),
        progress_reward_scale=config.get("progress_reward_scale", 10.0),
        lap_bonus=config.get("lap_bonus", 5.0),
        off_track_penalty=config.get("off_track_penalty", 5.0),
        collision_penalty=config.get("collision_penalty", 5.0),
        step_penalty=config.get("step_penalty", 0.0),
    )
    return env, track_name


def _select_action(agent, state, allow_idle_actions=True):
    if allow_idle_actions:
        return agent.act(state, training=False)

    allowed = np.array(sorted(ACCELERATING_ACTIONS), dtype=np.int64)
    q_values = agent.online_net.forward(state.reshape(1, -1)).flatten()
    return int(allowed[np.argmax(q_values[allowed])])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent in the RacingEnv.")
    parser.add_argument("--model-path", default="models/best_model.npz", help="Path to saved model parameters")
    parser.add_argument("--config", default=None,
                        help="Path to training config JSON. Defaults to config.json next to the model.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--save-dir", default="images", help="Directory to save rendered images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--track", choices=["procedural"], default=None,
                        help="Override track type from config")
    parser.add_argument("--track-seed", type=int, default=None,
                        help="Override procedural track seed from config")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI window)")
    parser.add_argument("--live", action="store_true",
                        help="Show the rollout in a live Matplotlib window")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS in live mode")
    parser.add_argument("--gif", "--save-gif", action="store_true",
                        help="Record and save GIF animation of each evaluation episode")
    parser.add_argument("--mp4", "--save-mp4", action="store_true",
                        help="Record and save MP4 video of each evaluation episode")
    parser.add_argument("--record-fps", type=int, default=30,
                        help="FPS for saved GIF/MP4 recordings")
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    config_path = args.config
    if config_path is None:
        candidate = os.path.join(os.path.dirname(args.model_path), "config.json")
        config_path = candidate if os.path.exists(candidate) else None
    config = _load_config(config_path) if config_path else {}

    if args.live:
        args.headless = False

    env, track_name = _make_env(args, config)
    allow_idle_actions = config.get("allow_idle_actions", True)

    print(f"Track type: {track_name}")
    print(f"Observation mode: {env.observation_mode}  Reward mode: {env.reward_mode}")
    print(f"Allow idle actions: {allow_idle_actions}")
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

    if env.observation_dim != agent.state_dim:
        raise ValueError(
            f"Model expects state_dim={agent.state_dim}, but evaluation env produces "
            f"{env.observation_dim}. Use the training config that matches this model."
        )

    renderer = MatplotlibRenderer(
        env.track,
        headless=args.headless,
        reward_line_progress=getattr(env, "_reward_line_progress", None),
    )

    total_rewards = []
    total_steps = []

    for ep in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + ep)
        ep_reward = 0.0

        if args.gif or args.mp4:
            renderer.start_recording()

        reward = 0.0
        info = {"lap_count": env.lap_count, "reward_lines_crossed": 0}
        for step in range(args.max_steps):
            action_idx = _select_action(agent, state, allow_idle_actions=allow_idle_actions)
            next_state, reward, done, info = env.step(ACTIONS[action_idx])
            ep_reward += reward
            renderer.render(
                env.state,
                step=step,
                reward=reward,
                total_reward=ep_reward,
                obstacles=env.obstacles,
                lap_count=info.get("lap_count"),
                reward_lines_crossed=info.get("reward_lines_crossed"),
            )
            if args.live and args.fps > 0:
                time.sleep(1.0 / args.fps)
            state = next_state
            if done:
                break

        total_rewards.append(ep_reward)
        total_steps.append(step + 1)
        print(f"ep={ep:2d}  reward={ep_reward:7.2f}  steps={step + 1:3d}")

        renderer.render(
            env.state,
            step=step,
            reward=reward,
            total_reward=ep_reward,
            obstacles=env.obstacles,
            lap_count=info.get("lap_count"),
            reward_lines_crossed=info.get("reward_lines_crossed"),
        )
        fig_path = os.path.join(args.save_dir, f"eval_ep{ep}_final.png")
        renderer.fig.savefig(fig_path, dpi=150)
        print(f"  Saved {fig_path}")

        if args.gif:
            gif_path = os.path.join(args.save_dir, f"eval_ep{ep}.gif")
            renderer.save_animation(gif_path, fps=args.record_fps)
            print(f"  Saved {gif_path}")

        if args.mp4:
            mp4_path = os.path.join(args.save_dir, f"eval_ep{ep}.mp4")
            renderer.save_video(mp4_path, fps=args.record_fps)
            print(f"  Saved {mp4_path}")

        if args.gif or args.mp4:
            renderer.stop_recording()

    if args.live:
        renderer.show()
    renderer.close()

    print(
        f"\nEvaluation over {args.episodes} episodes:\n"
        f"  Average reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}\n"
        f"  Average steps:  {np.mean(total_steps):.1f} +/- {np.std(total_steps):.1f}"
    )


if __name__ == "__main__":
    main()
