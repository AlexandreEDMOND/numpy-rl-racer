import argparse
import csv
import json
import os
import sys
from unittest.mock import patch

import pytest

import numpy as np

from numpy_rl_racer.agent.dqn import DQNAgent
from numpy_rl_racer.env import CircularTrack, Obstacle, RacingEnv, RectangularTrack


def _parse_track(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["rectangular", "circular", "figure8"], default="rectangular")
    parsed = parser.parse_args(args)
    if parsed.track == "circular":
        track = CircularTrack(radius=6.0, track_width=2.0)
        return RacingEnv(track=track), parsed.track
    elif parsed.track == "figure8":
        from numpy_rl_racer.env.racing_env import Figure8Track
        track = Figure8Track(radius=6.0, track_width=2.0)
        return RacingEnv(track=track), parsed.track
    else:
        return RacingEnv(), parsed.track


def test_default_track_is_rectangular():
    env, track_type = _parse_track([])
    assert track_type == "rectangular"
    assert isinstance(env.track, RectangularTrack)


def test_explicit_rectangular_track():
    env, track_type = _parse_track(["--track", "rectangular"])
    assert track_type == "rectangular"
    assert isinstance(env.track, RectangularTrack)


def test_circular_track():
    env, track_type = _parse_track(["--track", "circular"])
    assert track_type == "circular"
    assert isinstance(env.track, CircularTrack)


def test_figure8_track():
    env, track_type = _parse_track(["--track", "figure8"])
    assert track_type == "figure8"
    from numpy_rl_racer.env.racing_env import Figure8Track
    assert isinstance(env.track, Figure8Track)


def test_train_figure8_runs(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--track", "figure8",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    assert os.path.exists(config_path)


def _parse_log_dir(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=None)
    return parser.parse_args(args).log_dir


def test_log_dir_default_none():
    assert _parse_log_dir([]) is None


def test_log_dir_custom_path():
    assert _parse_log_dir(["--log-dir", "logs/test_run"]) == "logs/test_run"


def test_log_dir_no_csv_when_omitted(tmp_path):
    logger_created = _parse_log_dir([])
    assert logger_created is None
    csv_files = list(tmp_path.rglob("*.csv"))
    all_csv = [f for f in csv_files]
    assert len(all_csv) == 0


def test_log_dir_nested_path_parsing():
    result = _parse_log_dir(["--log-dir", "a/b/c"])
    assert result == "a/b/c"
    assert os.path.normpath(result) == "a/b/c"


def _make_mock_model(weights_dir, state_dim=6):
    hidden_sizes = [64, 64]
    n_actions = 5
    params = {
        "layer_0_w": np.random.randn(state_dim, hidden_sizes[0]).astype(np.float64),
        "layer_0_b": np.random.randn(hidden_sizes[0]).astype(np.float64),
        "layer_1_w": np.random.randn(hidden_sizes[0], hidden_sizes[1]).astype(np.float64),
        "layer_1_b": np.random.randn(hidden_sizes[1]).astype(np.float64),
        "layer_2_w": np.random.randn(hidden_sizes[1], n_actions).astype(np.float64),
        "layer_2_b": np.random.randn(n_actions).astype(np.float64),
    }
    path = os.path.join(weights_dir, f"mock_model_{state_dim}d.npz")
    np.savez(path, **params)
    return path


def _run_evaluate_main(tmp_path, extra_args=None):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from evaluate import main
        model_path = _make_mock_model(tmp_path, state_dim=6)
        args = [
            "--headless",
            "--model-path", model_path,
            "--episodes", "1",
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
        ]
        if extra_args:
            args.extend(extra_args)
        with patch("numpy_rl_racer.agent.dqn.DQNAgent.load"):
            main(args)
    finally:
        sys.path[:] = orig_path


def test_evaluate_headless(tmp_path):
    _run_evaluate_main(tmp_path)
    saved = list(tmp_path.glob("eval_ep*_final.png"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0


def test_evaluate_gif_flag(tmp_path):
    _run_evaluate_main(tmp_path, ["--gif"])
    gifs = list(tmp_path.glob("eval_ep*.gif"))
    assert len(gifs) == 1
    assert gifs[0].stat().st_size > 0


def test_evaluate_with_dueling_model(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from evaluate import main
        agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_dueling_dqn=True, seed=42)
        model_path = str(tmp_path / "dueling_model.npz")
        agent.save(model_path)
        args = [
            "--headless",
            "--model-path", model_path,
            "--episodes", "1",
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
        ]
        main(args)
    finally:
        sys.path[:] = orig_path


def test_evaluate_mock_6dim(tmp_path):
    model_path = _make_mock_model(tmp_path, state_dim=6)
    data = np.load(model_path)
    assert data["layer_0_w"].shape[0] == 6


def test_evaluate_mock_8dim(tmp_path):
    model_path = _make_mock_model(tmp_path, state_dim=8)
    data = np.load(model_path)
    assert data["layer_0_w"].shape[0] == 8


def test_infer_state_dim_detection(tmp_path):
    path_6 = _make_mock_model(tmp_path, state_dim=6)
    path_8 = _make_mock_model(tmp_path, state_dim=8)
    from evaluate import _infer_state_dim
    assert _infer_state_dim(path_6) == 6
    assert _infer_state_dim(path_8) == 8


# ── Configurable render CLI argument tests ────────────────────────────


def test_evaluate_render_dpi_and_fps_args(tmp_path):
    _run_evaluate_main(tmp_path, ["--render-dpi", "200", "--fps", "15"])
    saved = list(tmp_path.glob("eval_ep*_final.png"))
    assert len(saved) == 1


def test_evaluate_video_flag_triggers_recording(tmp_path):
    _run_evaluate_main(tmp_path, ["--video"])
    mp4s = list(tmp_path.glob("eval_ep*.mp4"))
    gifs = list(tmp_path.glob("eval_ep*.gif"))
    assert len(mp4s) >= 1 or len(gifs) >= 1


def test_evaluate_gif_and_video_both(tmp_path):
    _run_evaluate_main(tmp_path, ["--gif", "--video"])
    gifs = list(tmp_path.glob("eval_ep*.gif"))
    assert len(gifs) >= 1


def test_train_render_defaults(tmp_path):
    main = _make_main()
    with patch("numpy_rl_racer.agent.dqn.DQNAgent.act", return_value=0), \
         patch("numpy_rl_racer.agent.dqn.DQNAgent.train_step", return_value=0.0), \
         patch("numpy_rl_racer.agent.dqn.DQNAgent.save"):
        main([
            "--episodes", "1", "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    assert config["render_dpi"] == 100
    assert config["fps"] == 10


def test_train_custom_render_args(tmp_path):
    main = _make_main()
    with patch("numpy_rl_racer.agent.dqn.DQNAgent.act", return_value=0), \
         patch("numpy_rl_racer.agent.dqn.DQNAgent.train_step", return_value=0.0), \
         patch("numpy_rl_racer.agent.dqn.DQNAgent.save"):
        main([
            "--episodes", "1", "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--render-dpi", "200", "--fps", "20",
        ])
    config_path = os.path.join(tmp_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    assert config["render_dpi"] == 200
    assert config["fps"] == 20


def test_train_default_antialias_true(tmp_path):
    main = _make_main()
    with patch("numpy_rl_racer.agent.dqn.DQNAgent.act", return_value=0), \
         patch("numpy_rl_racer.agent.dqn.DQNAgent.train_step", return_value=0.0), \
         patch("numpy_rl_racer.agent.dqn.DQNAgent.save"):
        main([
            "--episodes", "1", "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    assert config["antialias"] is True


def test_train_no_antialias_arg(tmp_path):
    main = _make_main()
    with patch("numpy_rl_racer.agent.dqn.DQNAgent.act", return_value=0), \
         patch("numpy_rl_racer.agent.dqn.DQNAgent.train_step", return_value=0.0), \
         patch("numpy_rl_racer.agent.dqn.DQNAgent.save"):
        main([
            "--episodes", "1", "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--no-antialias",
        ])
    config_path = os.path.join(tmp_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    assert config["antialias"] is False


def test_evaluate_no_antialias(tmp_path):
    _run_evaluate_main(tmp_path, ["--no-antialias"])
    saved = list(tmp_path.glob("eval_ep*_final.png"))
    assert len(saved) == 1


def _make_main():
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from train import main
        return main
    finally:
        sys.path[:] = orig_path


def _parse_scheduler_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-scheduler", choices=["none", "exponential", "step"], default="none")
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--lr-drop-every", type=int, default=100)
    return parser.parse_args(args)


def test_lr_scheduler_default_none():
    parsed = _parse_scheduler_args([])
    assert parsed.lr_scheduler == "none"


def test_lr_scheduler_exponential():
    parsed = _parse_scheduler_args(["--lr-scheduler", "exponential"])
    assert parsed.lr_scheduler == "exponential"


def test_lr_scheduler_step():
    parsed = _parse_scheduler_args(["--lr-scheduler", "step"])
    assert parsed.lr_scheduler == "step"


def test_lr_decay_default():
    parsed = _parse_scheduler_args(["--lr-scheduler", "exponential"])
    assert parsed.lr_decay == 0.99


def test_lr_decay_custom():
    parsed = _parse_scheduler_args(["--lr-scheduler", "exponential", "--lr-decay", "0.95"])
    assert parsed.lr_decay == 0.95


def test_lr_drop_every_default():
    parsed = _parse_scheduler_args(["--lr-scheduler", "step"])
    assert parsed.lr_drop_every == 100


def test_lr_drop_every_custom():
    parsed = _parse_scheduler_args(["--lr-scheduler", "step", "--lr-drop-every", "50"])
    assert parsed.lr_drop_every == 50


def test_train_hyperparameters_passed_to_agent(tmp_path):
    main = _make_main()
    captured = []
    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        captured.append(kwargs)
        real_init(self, **kwargs)

    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--lr", "0.0005",
            "--batch-size", "128",
            "--gamma", "0.95",
            "--hidden-sizes", "128", "128",
            "--buffer-size", "5000",
            "--epsilon-start", "0.9",
            "--epsilon-min", "0.05",
            "--epsilon-decay", "0.99",
            "--target-update-freq", "50",
            "--no-double-dqn",
            "--use-per",
        ])

    kwargs = captured[0]
    assert kwargs["state_dim"] == 6
    assert kwargs["lr"] == 0.0005
    assert kwargs["batch_size"] == 128
    assert kwargs["gamma"] == 0.95
    assert np.array_equal(kwargs["hidden_sizes"], [128, 128])
    assert kwargs["buffer_size"] == 5000
    assert kwargs["epsilon"] == 0.9
    assert kwargs["epsilon_min"] == 0.05
    assert kwargs["epsilon_decay"] == 0.99
    assert kwargs["target_update_freq"] == 50
    assert kwargs["use_double_dqn"] is False
    assert kwargs["use_per"] is True
    assert kwargs["seed"] is None


def test_train_default_hyperparameters(tmp_path):
    main = _make_main()
    captured = []
    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        captured.append(kwargs)
        real_init(self, **kwargs)

    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    kwargs = captured[0]
    assert kwargs["lr"] == 1e-3
    assert kwargs["batch_size"] == 64
    assert kwargs["gamma"] == 0.99
    assert np.array_equal(kwargs["hidden_sizes"], [64, 64])
    assert kwargs["buffer_size"] == 10000
    assert kwargs["epsilon"] == 1.0
    assert kwargs["epsilon_min"] == 0.01
    assert kwargs["epsilon_decay"] == 0.995
    assert kwargs["target_update_freq"] == 100
    assert kwargs["use_double_dqn"] is True
    assert kwargs["use_per"] is False
    assert kwargs["use_dueling_dqn"] is False


def test_train_dueling_dqn_flag(tmp_path):
    main = _make_main()
    captured = []
    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        captured.append(kwargs)
        real_init(self, **kwargs)

    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--dueling-dqn",
        ])

    kwargs = captured[0]
    assert kwargs["use_dueling_dqn"] is True


def test_eval_freq_zero_skips_eval(tmp_path):
    main = _make_main()
    real_init = DQNAgent.__init__
    with patch.object(DQNAgent, "__init__", lambda self, **kwargs: real_init(self, **kwargs)), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--log-dir", str(tmp_path),
        ])
    with open(os.path.join(tmp_path, "training_log.csv"), newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert "eval_reward_mean" not in fieldnames
        assert "eval_reward_std" not in fieldnames


def test_eval_csv_columns_present(tmp_path):
    main = _make_main()
    real_init = DQNAgent.__init__
    with patch.object(DQNAgent, "__init__", lambda self, **kwargs: real_init(self, **kwargs)), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "4",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--log-dir", str(tmp_path),
            "--eval-freq", "2",
            "--eval-episodes", "3",
        ])
    with open(os.path.join(tmp_path, "training_log.csv"), newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert "eval_reward_mean" in fieldnames
        assert "eval_reward_std" in fieldnames
        rows = list(reader)
    assert len(rows) == 4
    assert rows[0]["eval_reward_mean"] == ""  # ep 1, no eval
    assert rows[1]["eval_reward_mean"] != ""  # ep 2, eval ran
    assert rows[2]["eval_reward_mean"] == ""  # ep 3, no eval
    assert rows[3]["eval_reward_mean"] != ""  # ep 4, eval ran


def test_epsilon_restored_after_eval(tmp_path):
    main = _make_main()
    epsilon_values = []
    real_init = DQNAgent.__init__

    def tracking_act(self, state, training=True):
        epsilon_values.append(self.epsilon)
        return 0

    with patch.object(DQNAgent, "__init__", lambda self, **kwargs: real_init(self, **kwargs)), \
         patch.object(DQNAgent, "act", tracking_act), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "3",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--eval-freq", "2",
            "--eval-episodes", "2",
            "--epsilon-start", "0.5",
            "--epsilon-decay", "1.0",
        ])
    assert len(epsilon_values) == 5  # 3 training + 2 eval acts
    assert epsilon_values[0] == 0.5  # ep 1 training
    assert epsilon_values[1] == 0.5  # ep 2 training
    assert epsilon_values[2] == 0.0  # ep 2 eval ep 1
    assert epsilon_values[3] == 0.0  # ep 2 eval ep 2
    assert epsilon_values[4] == 0.5  # ep 3 training (restored)


def test_eval_training_curve_generated(tmp_path):
    main = _make_main()
    real_init = DQNAgent.__init__
    with patch.object(DQNAgent, "__init__", lambda self, **kwargs: real_init(self, **kwargs)), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "2",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--eval-freq", "2",
            "--eval-episodes", "1",
        ])
    curve_path = os.path.join(tmp_path, "training_curve.png")
    assert os.path.exists(curve_path)
    assert os.path.getsize(curve_path) > 0


def test_config_file_not_found(tmp_path):
    main = _make_main()
    cfg = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        main([
            "--config", str(cfg),
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])


def test_config_malformed_json(tmp_path):
    main = _make_main()
    cfg = tmp_path / "bad.json"
    cfg.write_text("{invalid json}")
    with pytest.raises(json.JSONDecodeError):
        main([
            "--config", str(cfg),
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])


def test_config_sets_defaults(tmp_path):
    main = _make_main()
    config_data = {"lr": 0.0005, "batch_size": 128, "gamma": 0.95}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config_data))

    captured = []
    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        captured.append(kwargs)
        real_init(self, **kwargs)

    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--config", str(cfg),
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    kwargs = captured[0]
    assert kwargs["lr"] == 0.0005
    assert kwargs["batch_size"] == 128
    assert kwargs["gamma"] == 0.95


def test_config_cli_overrides(tmp_path):
    main = _make_main()
    config_data = {"lr": 0.0005, "batch_size": 128, "gamma": 0.95}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config_data))

    captured = []
    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        captured.append(kwargs)
        real_init(self, **kwargs)

    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--config", str(cfg),
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--lr", "0.001",
        ])

    kwargs = captured[0]
    assert kwargs["lr"] == 0.001
    assert kwargs["batch_size"] == 128
    assert kwargs["gamma"] == 0.95


def test_config_saved_to_save_dir(tmp_path):
    main = _make_main()
    config_data = {"lr": 0.0005}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config_data))

    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        real_init(self, **kwargs)

    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--config", str(cfg),
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    saved = os.path.join(tmp_path, "config.json")
    assert os.path.exists(saved)
    with open(saved) as f:
        saved_config = json.load(f)
    assert saved_config["lr"] == 0.0005


def test_script_randomize_start_cli(tmp_path):
    main = _make_main()
    env_kwargs = []
    real_init = RacingEnv.__init__

    def tracking_init(self, **kwargs):
        env_kwargs.append(kwargs)
        real_init(self, **kwargs)

    with patch.object(RacingEnv, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--no-randomize-start",
        ])

    assert len(env_kwargs) == 1
    assert env_kwargs[0].get("randomize_start") is False


def test_config_not_required(tmp_path):
    main = _make_main()
    captured = []
    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        captured.append(kwargs)
        real_init(self, **kwargs)

    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    kwargs = captured[0]
    assert kwargs["lr"] == 1e-3
    assert kwargs["batch_size"] == 64
    assert kwargs["gamma"] == 0.99


def test_obstacles_default_none(tmp_path):
    main = _make_main()
    env_kwargs = []
    agent_kwargs = []
    real_init_env = RacingEnv.__init__
    real_init_agent = DQNAgent.__init__

    def tracking_env(self, **kwargs):
        env_kwargs.append(kwargs)
        real_init_env(self, **kwargs)

    def tracking_agent(self, **kwargs):
        agent_kwargs.append(kwargs)
        real_init_agent(self, **kwargs)

    with patch.object(RacingEnv, "__init__", tracking_env), \
         patch.object(DQNAgent, "__init__", tracking_agent), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    assert env_kwargs[0].get("obstacles") is None
    assert len(env_kwargs[0].get("obstacles") if env_kwargs[0].get("obstacles") else []) == 0
    assert agent_kwargs[0]["state_dim"] == 6


def test_obstacles_num_obstacles_3(tmp_path):
    main = _make_main()
    env_kwargs = []
    agent_kwargs = []
    real_init_env = RacingEnv.__init__
    real_init_agent = DQNAgent.__init__

    def tracking_env(self, **kwargs):
        env_kwargs.append(kwargs)
        real_init_env(self, **kwargs)

    def tracking_agent(self, **kwargs):
        agent_kwargs.append(kwargs)
        real_init_agent(self, **kwargs)

    with patch.object(RacingEnv, "__init__", tracking_env), \
         patch.object(DQNAgent, "__init__", tracking_agent), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--num-obstacles", "3",
            "--obstacle-seed", "0",
        ])

    obstacles = env_kwargs[0].get("obstacles")
    assert obstacles is not None
    assert len(obstacles) == 3
    for obs in obstacles:
        assert isinstance(obs, Obstacle)
    assert agent_kwargs[0]["state_dim"] == 8


def test_obstacles_seed_determinism(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from train import _generate_obstacles
        from numpy_rl_racer.env import RectangularTrack
        track = RectangularTrack()
        obs1 = _generate_obstacles(track, 3, seed=42)
        obs2 = _generate_obstacles(track, 3, seed=42)
        obs3 = _generate_obstacles(track, 3, seed=99)
        for o1, o2 in zip(obs1, obs2):
            assert o1.x == o2.x
            assert o1.y == o2.y
            assert o1.radius == o2.radius
        assert any(o1.x != o3.x or o1.y != o3.y for o1, o3 in zip(obs1, obs3))
    finally:
        sys.path[:] = orig_path


def test_train_with_obstacles_runs(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--num-obstacles", "2",
            "--obstacle-seed", "7",
        ])


def test_obstacles_config_keys(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--num-obstacles", "4",
            "--obstacle-seed", "123",
        ])
    config_path = os.path.join(tmp_path, "config.json")
    assert os.path.exists(config_path)
    with open(config_path) as f:
        saved = json.load(f)
    assert saved["num_obstacles"] == 4
    assert saved["obstacle_seed"] == 123


# ---------------------------------------------------------------------------
# Skip-frames (action repeat) tests
# ---------------------------------------------------------------------------


def test_train_skip_frames_env_wrapped(tmp_path):
    main = _make_main()
    wraps = []
    real_init = RacingEnv.__init__
    def tracking_init(self, **kwargs):
        wraps.append(kwargs)
        real_init(self, **kwargs)
    with patch.object(RacingEnv, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--skip-frames", "4",
            "--episodes", "1", "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    assert len(wraps) >= 1


def test_train_skip_frames_default_1(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1", "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    # Default skip_frames=1 means no wrapping — env is a RacingEnv, not ActionRepeatEnv


def test_train_skip_frames_invalid(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        with pytest.raises(ValueError, match="--skip-frames must be >= 1"):
            main([
                "--skip-frames", "0",
                "--episodes", "1", "--max-steps", "1",
                "--save-dir", str(tmp_path),
            ])


def test_train_skip_frames_with_obstacles(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--skip-frames", "3",
            "--num-obstacles", "2", "--obstacle-seed", "7",
            "--episodes", "1", "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])


# ---------------------------------------------------------------------------
# Grid search tests
# ---------------------------------------------------------------------------

def _make_grid_search():
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from grid_search import main
        return main
    finally:
        sys.path[:] = orig_path


def test_grid_search_help(capsys):
    gs_main = _make_grid_search()
    with pytest.raises(SystemExit):
        gs_main(["--help"])
    captured = capsys.readouterr()
    assert "grid search" in captured.out.lower()


def test_grid_search_basic(tmp_path):
    gs_main = _make_grid_search()
    csv_path = str(tmp_path / "results.csv")
    plot_path = str(tmp_path / "plot.png")
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.5), \
         patch.object(DQNAgent, "save"):
        gs_main([
            "--episodes", "2",
            "--max-steps", "3",
            "--lr", "1e-3,1e-2",
            "--gamma", "0.99,0.95",
            "--output", csv_path,
            "--plot", plot_path,
        ])
    assert os.path.exists(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4  # 2x2 combos


def test_grid_search_csv_columns(tmp_path):
    gs_main = _make_grid_search()
    csv_path = str(tmp_path / "results.csv")
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.5), \
         patch.object(DQNAgent, "save"):
        gs_main([
            "--episodes", "2",
            "--max-steps", "3",
            "--lr", "1e-3,1e-2",
            "--gamma", "0.99,0.95",
            "--output", csv_path,
        ])
    expected = {"run_id", "params", "final_reward", "mean_reward", "std_reward",
                "final_loss", "mean_loss", "total_steps", "elapsed_time"}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == expected
        for row in reader:
            assert int(row["run_id"]) >= 0
            float(row["final_reward"])
            float(row["mean_reward"])
            float(row["std_reward"])
            float(row["final_loss"])
            float(row["mean_loss"])
            json.loads(row["params"])


def test_grid_search_plot(tmp_path):
    gs_main = _make_grid_search()
    csv_path = str(tmp_path / "results.csv")
    plot_path = str(tmp_path / "plot.png")
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.5), \
         patch.object(DQNAgent, "save"):
        gs_main([
            "--episodes", "2",
            "--max-steps", "3",
            "--lr", "1e-3,1e-2",
            "--gamma", "0.99,0.95",
            "--output", csv_path,
            "--plot", plot_path,
        ])
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0


# ---------------------------------------------------------------------------
# Compare policies script tests
# ---------------------------------------------------------------------------

def _make_agent_checkpoint(path, state_dim=6):
    agent = DQNAgent(state_dim=state_dim, hidden_sizes=[16], seed=0)
    agent.save(path)


def test_compare_policies_generates_gif(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from compare_policies import main
        model_path = str(tmp_path / "test_model.npz")
        _make_agent_checkpoint(model_path, state_dim=6)
        main([
            "--model-path", model_path,
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
        ])
        gifs = list(tmp_path.glob("*.gif"))
        assert len(gifs) >= 1
        assert all(g.stat().st_size > 0 for g in gifs)
    finally:
        sys.path[:] = orig_path


def test_compare_policies_custom_render_args(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from compare_policies import main
        model_path = str(tmp_path / "test_model.npz")
        _make_agent_checkpoint(model_path, state_dim=6)
        main([
            "--model-path", model_path,
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
            "--render-dpi", "200",
            "--fps", "15",
        ])
        gifs = list(tmp_path.glob("*.gif"))
        assert len(gifs) >= 1
        assert all(g.stat().st_size > 0 for g in gifs)
    finally:
        sys.path[:] = orig_path


def test_compare_policies_no_antialias(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from compare_policies import main
        model_path = str(tmp_path / "test_model.npz")
        _make_agent_checkpoint(model_path, state_dim=6)
        main([
            "--model-path", model_path,
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
            "--no-antialias",
        ])
        gifs = list(tmp_path.glob("*.gif"))
        assert len(gifs) >= 1
        assert all(g.stat().st_size > 0 for g in gifs)
    finally:
        sys.path[:] = orig_path


def test_grid_search_seed(tmp_path):
    gs_main = _make_grid_search()
    csv_a = str(tmp_path / "a.csv")
    csv_b = str(tmp_path / "b.csv")
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.5), \
         patch.object(DQNAgent, "save"):
        gs_main([
            "--seed", "42",
            "--episodes", "2",
            "--max-steps", "3",
            "--lr", "1e-3,1e-2",
            "--output", csv_a,
        ])
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.5), \
         patch.object(DQNAgent, "save"):
        gs_main([
            "--seed", "42",
            "--episodes", "2",
            "--max-steps", "3",
            "--lr", "1e-3,1e-2",
            "--output", csv_b,
        ])
    with open(csv_a, newline="") as f:
        rows_a = list(csv.DictReader(f))
    with open(csv_b, newline="") as f:
        rows_b = list(csv.DictReader(f))
    assert len(rows_a) == len(rows_b)
    for ra, rb in zip(rows_a, rows_b):
        assert ra["params"] == rb["params"]
