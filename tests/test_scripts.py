import argparse
import csv
import json
import os
import sys
import types
from unittest.mock import patch

import pytest

import numpy as np

from numpy_rl_racer.agent.dqn import DQNAgent
from numpy_rl_racer.env import Obstacle, ProceduralTrack, RacingEnv


def _parse_track(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["procedural"], default="procedural")
    parsed = parser.parse_args(args)
    return RacingEnv(track=ProceduralTrack(seed=0)), parsed.track


def test_default_track_is_procedural():
    env, track_type = _parse_track([])
    assert track_type == "procedural"
    assert isinstance(env.track, ProceduralTrack)


def _make_preview_tracks():
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from preview_tracks import main
        return main
    finally:
        sys.path[:] = orig_path


def test_preview_tracks_generates_nonblank_gallery(tmp_path):
    preview_main = _make_preview_tracks()
    output = tmp_path / "gallery.png"
    preview_main(["--seeds", "0", "1", "--cols", "2", "--output", str(output)])

    assert output.exists()
    from PIL import Image
    with Image.open(output) as img:
        pixels = np.asarray(img.convert("RGB"))
    assert pixels.std() > 0


def test_train_procedural_runs(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--track", "procedural",
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


def test_evaluate_metrics_only_multi_seed_no_png(tmp_path):
    _run_evaluate_main(tmp_path, [
        "--metrics-only", "--track-seeds", "0", "1",
        "--episodes", "1", "--max-steps", "3",
    ])
    assert not list(tmp_path.glob("*eval_ep*_final.png"))
    assert not list(tmp_path.glob("*.gif"))
    assert not list(tmp_path.glob("*.mp4"))
    summary = tmp_path / "eval_summary.csv"
    assert summary.exists()
    with open(summary, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_evaluate_metrics_only_single_seed_no_png(tmp_path):
    _run_evaluate_main(tmp_path, [
        "--metrics-only", "--episodes", "2", "--max-steps", "3",
    ])
    assert not list(tmp_path.glob("*eval_ep*_final.png"))
    assert not list(tmp_path.glob("*.gif"))
    assert not list(tmp_path.glob("*.mp4"))
    summary = tmp_path / "eval_summary.csv"
    assert summary.exists()
    with open(summary, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1


def test_evaluate_metrics_only_with_gif_skips_recording(tmp_path, capsys):
    _run_evaluate_main(tmp_path, [
        "--metrics-only", "--gif", "--track-seeds", "0",
        "--episodes", "1", "--max-steps", "3",
    ])
    assert not list(tmp_path.glob("*.gif"))
    assert not list(tmp_path.glob("*.mp4"))
    assert (tmp_path / "eval_summary.csv").exists()
    captured = capsys.readouterr()
    assert "--metrics-only set: skipping GIF/MP4 recording" in captured.out


def test_evaluate_gif_flag(tmp_path):
    _run_evaluate_main(tmp_path, ["--gif"])
    gifs = list(tmp_path.glob("eval_ep*.gif"))
    assert len(gifs) == 1
    assert gifs[0].stat().st_size > 0


def test_evaluate_mp4_flag_calls_video_export(tmp_path):
    with patch(
        "numpy_rl_racer.rendering.matplotlib_renderer.MatplotlibRenderer.save_video"
    ) as save_video:
        _run_evaluate_main(tmp_path, ["--mp4", "--record-fps", "24"])
    save_video.assert_called_once()
    assert save_video.call_args.kwargs["fps"] == 24


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


def test_evaluate_multi_track_seeds_runs(tmp_path):
    _run_evaluate_main(tmp_path, ["--track-seeds", "0", "1", "--episodes", "1", "--max-steps", "3"])
    summary = tmp_path / "eval_summary.csv"
    assert summary.exists()


def test_evaluate_summary_csv_one_row_per_seed(tmp_path):
    _run_evaluate_main(tmp_path, ["--track-seeds", "0", "1", "2", "--episodes", "2"])
    summary = tmp_path / "eval_summary.csv"
    assert summary.exists()
    with open(summary, newline="") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == {
            "seed", "episodes", "mean_reward", "std_reward",
            "mean_steps", "std_steps", "laps_completed_total",
        }
        rows = list(reader)
    assert len(rows) == 3
    assert [int(r["seed"]) for r in rows] == [0, 1, 2]
    assert all(int(r["episodes"]) == 2 for r in rows)


def test_evaluate_multi_seed_summary_plot_written(tmp_path):
    _run_evaluate_main(tmp_path, [
        "--track-seeds", "0", "1", "2", "--episodes", "2", "--summary-plot",
    ])
    plot = tmp_path / "eval_summary.png"
    assert plot.exists()
    assert plot.stat().st_size > 0
    summary = tmp_path / "eval_summary.csv"
    assert summary.exists()
    with open(summary, newline="") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == {
            "seed", "episodes", "mean_reward", "std_reward",
            "mean_steps", "std_steps", "laps_completed_total",
        }
        rows = list(reader)
    assert len(rows) == 3


def test_evaluate_multi_seed_summary_plot_default_off(tmp_path):
    _run_evaluate_main(tmp_path, [
        "--track-seeds", "0", "1", "--episodes", "1",
    ])
    plot = tmp_path / "eval_summary.png"
    assert not plot.exists()
    assert (tmp_path / "eval_summary.csv").exists()


def test_evaluate_multi_seed_summary_plot_headless(tmp_path):
    _run_evaluate_main(tmp_path, [
        "--track-seeds", "0", "1", "--episodes", "1",
        "--summary-plot", "--headless",
    ])
    plot = tmp_path / "eval_summary.png"
    assert plot.exists()
    assert plot.stat().st_size > 0


def test_evaluate_single_seed_no_summary_plot(tmp_path):
    _run_evaluate_main(tmp_path, ["--summary-plot"])
    plot = tmp_path / "eval_summary.png"
    assert not plot.exists()


def test_evaluate_multi_track_seeds_dim_mismatch(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from evaluate import main
        model_path = _make_mock_model(tmp_path, state_dim=6)
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"num_obstacles": 3}))
        args = [
            "--headless",
            "--model-path", model_path,
            "--config", str(cfg),
            "--track-seeds", "0",
            "--episodes", "1",
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
        ]
        with patch("numpy_rl_racer.agent.dqn.DQNAgent.load"):
            with pytest.raises(ValueError, match="observation_dim"):
                main(args)
    finally:
        sys.path[:] = orig_path


def test_evaluate_single_track_unchanged(tmp_path):
    _run_evaluate_main(tmp_path)
    saved = list(tmp_path.glob("eval_ep*_final.png"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0


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


def _parse_optimizer_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    return parser.parse_args(args)


def test_train_optimizer_default_sgd():
    parsed = _parse_optimizer_args([])
    assert parsed.optimizer == "sgd"
    assert parsed.adam_beta1 == 0.9
    assert parsed.adam_beta2 == 0.999
    assert parsed.adam_eps == 1e-8


def test_train_optimizer_adam_args_parsed():
    parsed = _parse_optimizer_args([
        "--optimizer", "adam",
        "--adam-beta1", "0.8",
        "--adam-beta2", "0.99",
        "--adam-eps", "1e-7",
    ])
    assert parsed.optimizer == "adam"
    assert parsed.adam_beta1 == 0.8
    assert parsed.adam_beta2 == 0.99
    assert parsed.adam_eps == 1e-7


def test_train_optimizer_invalid_rejected():
    with pytest.raises(SystemExit):
        _parse_optimizer_args(["--optimizer", "rmsprop"])


def test_train_optimizer_adam_runs(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0):
        main([
            "--optimizer", "adam",
            "--episodes", "3",
            "--max-steps", "10",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    assert os.path.exists(config_path)
    with open(config_path) as f:
        cfg = json.load(f)
    assert cfg["optimizer"] == "adam"
    assert cfg["adam_beta1"] == 0.9
    assert cfg["adam_beta2"] == 0.999
    final_model = os.path.join(tmp_path, "final_model.npz")
    assert os.path.exists(final_model)
    best_model = os.path.join(tmp_path, "best_model.npz")
    assert os.path.exists(best_model)
    curve = os.path.join(tmp_path, "training_curve.png")
    assert os.path.exists(curve)


def test_train_optimizer_adam_passed_to_agent(tmp_path):
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
            "--optimizer", "adam",
            "--adam-beta1", "0.8",
            "--adam-beta2", "0.99",
            "--adam-eps", "1e-7",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    kwargs = captured[0]
    assert kwargs["optimizer"] == "adam"
    assert kwargs["betas"] == (0.8, 0.99)
    assert kwargs["eps"] == 1e-7


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
    assert kwargs["state_dim"] == 9
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


def test_train_per_hyperparameters_passed_to_agent(tmp_path):
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
            "--use-per",
            "--per-alpha", "0.5",
            "--per-beta0", "0.6",
            "--per-beta-anneal-steps", "1000",
        ])

    kwargs = captured[0]
    assert kwargs["use_per"] is True
    assert kwargs["alpha"] == 0.5
    assert kwargs["beta0"] == 0.6
    assert kwargs["beta_anneal_steps"] == 1000


def test_train_per_hyperparameters_default_when_use_per(tmp_path):
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
            "--use-per",
        ])

    kwargs = captured[0]
    assert kwargs["use_per"] is True
    assert kwargs["alpha"] == 0.6
    assert kwargs["beta0"] == 0.4
    assert kwargs["beta_anneal_steps"] == 100000


def test_train_per_hyperparameters_round_trip_via_config(tmp_path):
    main = _make_main()
    captured = []
    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        captured.append(kwargs)
        real_init(self, **kwargs)

    save_dir = tmp_path / "first"
    save_dir.mkdir()
    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(save_dir),
            "--use-per",
            "--per-alpha", "0.5",
            "--per-beta0", "0.6",
            "--per-beta-anneal-steps", "1000",
        ])

    assert captured[0]["alpha"] == 0.5
    assert captured[0]["beta0"] == 0.6
    assert captured[0]["beta_anneal_steps"] == 1000

    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["per_alpha"] == 0.5
    assert cfg["per_beta0"] == 0.6
    assert cfg["per_beta_anneal_steps"] == 1000
    assert cfg["use_per"] is True

    captured.clear()
    reloaded_dir = tmp_path / "second"
    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--config", os.path.join(save_dir, "config.json"),
            "--use-per",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(reloaded_dir),
        ])

    kwargs = captured[0]
    assert kwargs["use_per"] is True
    assert kwargs["alpha"] == 0.5
    assert kwargs["beta0"] == 0.6
    assert kwargs["beta_anneal_steps"] == 1000


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
    assert kwargs["epsilon_decay"] == 0.9995
    assert kwargs["target_update_freq"] == 100
    assert kwargs["use_double_dqn"] is False
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
            "--allow-idle-actions",
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


def _read_training_log(tmp_path):
    log_path = os.path.join(tmp_path, "training_log.csv")
    assert os.path.exists(log_path), "training_log.csv was not written"
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return fieldnames, rows


def test_train_csv_includes_progress_and_laps_columns(tmp_path):
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
    fieldnames, _ = _read_training_log(tmp_path)
    assert "mean_progress" in fieldnames
    assert "laps_completed" in fieldnames


def test_train_csv_progress_columns_appended_after_collision(tmp_path):
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
            "--eval-freq", "1",
            "--eval-episodes", "1",
        ])
    fieldnames, _ = _read_training_log(tmp_path)
    assert fieldnames.index("collision_steps") < fieldnames.index("mean_progress")
    assert fieldnames.index("mean_progress") < fieldnames.index("laps_completed")
    assert fieldnames.index("laps_completed") < fieldnames.index("eval_reward_mean")
    # Regression: existing column still in its prior relative position.
    assert fieldnames.index("off_track_steps") < fieldnames.index("collision_steps")


def test_train_progress_column_numeric_value(tmp_path):
    main = _make_main()
    real_init = DQNAgent.__init__
    with patch.object(DQNAgent, "__init__", lambda self, **kwargs: real_init(self, **kwargs)), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "2",
            "--save-dir", str(tmp_path),
            "--log-dir", str(tmp_path),
        ])
    _, rows = _read_training_log(tmp_path)
    assert len(rows) == 1
    progress = float(rows[0]["mean_progress"])
    assert 0.0 <= progress <= 1.0
    laps = int(rows[0]["laps_completed"])
    assert laps >= 0


def test_train_curve_has_four_panels(tmp_path):
    main = _make_main()
    real_init = DQNAgent.__init__
    captured = {}

    import matplotlib.pyplot as plt

    real_subplots = plt.subplots

    def tracking_subplots(nrows=1, ncols=1, **kwargs):
        captured["nrows"] = nrows
        return real_subplots(nrows, ncols, **kwargs)

    with patch.object(DQNAgent, "__init__", lambda self, **kwargs: real_init(self, **kwargs)), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"), \
         patch.object(plt, "subplots", tracking_subplots):
        main([
            "--episodes", "3",
            "--max-steps", "2",
            "--save-dir", str(tmp_path),
        ])
    curve_path = os.path.join(tmp_path, "training_curve.png")
    assert os.path.exists(curve_path)
    assert os.path.getsize(curve_path) > 0
    assert captured.get("nrows") == 4


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


def test_script_randomize_start_enabled_cli(tmp_path):
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
            "--randomize-start",
        ])

    assert len(env_kwargs) == 1
    assert env_kwargs[0].get("randomize_start") is True


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
    assert agent_kwargs[0]["state_dim"] == 9


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
    assert agent_kwargs[0]["state_dim"] == 9


def test_obstacles_seed_determinism(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from train import _generate_obstacles
        track = ProceduralTrack(seed=0)
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
# Multi-track pool (--track-seeds) tests
# ---------------------------------------------------------------------------

def test_track_seeds_exported_from_env():
    from numpy_rl_racer.env import TrackPoolEnv
    assert TrackPoolEnv is not None


# ---------------------------------------------------------------------------
# Local ray angles / lidar range configuration tests
# ---------------------------------------------------------------------------

def test_train_local_ray_angles_passed_to_env(tmp_path):
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
            "--local-ray-angles", "-1.5708", "0", "1.5708",
        ])

    assert "local_ray_angles" in env_kwargs[0]
    assert len(env_kwargs[0]["local_ray_angles"]) == 3
    assert agent_kwargs[0]["state_dim"] == 7  # 4 base + 3 rays


def test_train_local_ray_angles_default_none_not_passed(tmp_path):
    main = _make_main()
    env_kwargs = []
    real_init_env = RacingEnv.__init__

    def tracking_env(self, **kwargs):
        env_kwargs.append(kwargs)
        real_init_env(self, **kwargs)

    with patch.object(RacingEnv, "__init__", tracking_env), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    assert "local_ray_angles" not in env_kwargs[0]


def test_train_local_ray_angles_persisted_to_config(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--local-ray-angles", "-1.5708", "-0.7854", "0", "0.7854", "1.5708",
        ])
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["local_ray_angles"] == [-1.5708, -0.7854, 0.0, 0.7854, 1.5708]


def test_train_local_ray_angles_default_null_in_config(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["local_ray_angles"] is None
    assert cfg["lidar_max_range"] is None


def test_train_lidar_max_range_passed_and_persisted(tmp_path):
    main = _make_main()
    env_kwargs = []
    real_init_env = RacingEnv.__init__

    def tracking_env(self, **kwargs):
        env_kwargs.append(kwargs)
        real_init_env(self, **kwargs)

    with patch.object(RacingEnv, "__init__", tracking_env), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--lidar-max-range", "15.0",
        ])

    assert env_kwargs[0]["lidar_max_range"] == 15.0
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["lidar_max_range"] == 15.0


def test_train_local_ray_angles_seven_angles_state_dim_eleven(tmp_path):
    main = _make_main()
    agent_kwargs = []
    real_init_agent = DQNAgent.__init__

    def tracking_agent(self, **kwargs):
        agent_kwargs.append(kwargs)
        real_init_agent(self, **kwargs)

    with patch.object(DQNAgent, "__init__", tracking_agent), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--local-ray-angles", "-1.5", "-1.0", "-0.5", "0", "0.5", "1.0", "1.5",
        ])

    assert agent_kwargs[0]["state_dim"] == 11  # 4 base + 7 rays


def test_train_track_seeds_forward_ray_angles(tmp_path):
    main = _make_main()
    env_kwargs = []
    real_init_env = RacingEnv.__init__

    def tracking_env(self, **kwargs):
        env_kwargs.append(kwargs)
        real_init_env(self, **kwargs)

    with patch.object(RacingEnv, "__init__", tracking_env), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--track-seeds", "0", "1",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--local-ray-angles", "-1.5708", "0", "1.5708",
            "--lidar-max-range", "15.0",
        ])

    assert all("local_ray_angles" in kw for kw in env_kwargs)
    assert all(len(kw["local_ray_angles"]) == 3 for kw in env_kwargs)
    assert all(kw["lidar_max_range"] == 15.0 for kw in env_kwargs)


def test_evaluate_reads_local_ray_angles_from_config(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from evaluate import main
        model_path = _make_mock_model(tmp_path, state_dim=7)
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "observation_mode": "local",
            "local_ray_angles": [-1.5708, 0.0, 1.5708],
        }))
        args = [
            "--headless",
            "--model-path", model_path,
            "--config", str(cfg),
            "--episodes", "1",
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
        ]
        with patch("numpy_rl_racer.agent.dqn.DQNAgent.load"):
            main(args)
    finally:
        sys.path[:] = orig_path
    saved = list(tmp_path.glob("eval_ep*_final.png"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0


def test_evaluate_reads_lidar_max_range_from_config(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from evaluate import main
        model_path = _make_mock_model(tmp_path, state_dim=9)
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "observation_mode": "local",
            "local_ray_angles": [-1.5708, 0.0, 1.5708, 2.0, 3.0],
            "lidar_max_range": 15.0,
        }))
        args = [
            "--headless",
            "--model-path", model_path,
            "--config", str(cfg),
            "--episodes", "1",
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
        ]
        with patch("numpy_rl_racer.agent.dqn.DQNAgent.load"):
            main(args)
    finally:
        sys.path[:] = orig_path
    saved = list(tmp_path.glob("eval_ep*_final.png"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0


# ---------------------------------------------------------------------------
# Checkpoint snapshots (--checkpoint-freq) tests
# ---------------------------------------------------------------------------

def test_train_no_checkpoint_freq_by_default(tmp_path):
    main = _make_main()
    saved_paths = []
    real_save = DQNAgent.save

    def tracking_save(self, path):
        saved_paths.append(path)
        real_save(self, path)

    with patch.object(DQNAgent, "save", tracking_save), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0):
        main([
            "--episodes", "3",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    assert not any("checkpoint_ep" in os.path.basename(p) for p in saved_paths)
    assert not list(tmp_path.glob("checkpoint_ep*.npz"))


def test_train_writes_periodic_checkpoints(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0):
        main([
            "--episodes", "4",
            "--max-steps", "1",
            "--checkpoint-freq", "2",
            "--save-dir", str(tmp_path),
        ])
    assert (tmp_path / "checkpoint_ep2.npz").exists()
    assert (tmp_path / "checkpoint_ep4.npz").exists()
    assert not (tmp_path / "checkpoint_ep1.npz").exists()
    assert not (tmp_path / "checkpoint_ep3.npz").exists()


def test_train_config_records_checkpoint_freq(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "2",
            "--max-steps", "1",
            "--checkpoint-freq", "2",
            "--save-dir", str(tmp_path),
        ])
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["checkpoint_freq"] == 2


def test_train_checkpoint_freq_round_trips_to_load(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0):
        main([
            "--episodes", "2",
            "--max-steps", "1",
            "--checkpoint-freq", "2",
            "--save-dir", str(tmp_path),
        ])
    checkpoint = tmp_path / "checkpoint_ep2.npz"
    assert checkpoint.exists()
    agent = DQNAgent(state_dim=9, hidden_sizes=[64, 64], seed=0)
    agent.load(str(checkpoint))  # should not raise


def test_train_checkpoint_freq_zero_default_in_config(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["checkpoint_freq"] == 0


def test_train_track_seeds_runs(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0):
        main([
            "--track-seeds", "0", "1", "2",
            "--episodes", "2",
            "--max-steps", "2",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    assert os.path.exists(config_path)
    with open(config_path) as f:
        cfg = json.load(f)
    assert cfg["track_seeds"] == [0, 1, 2]
    assert cfg["track_pool_mode"] == "round_robin"
    assert os.path.exists(os.path.join(tmp_path, "final_model.npz"))


def test_train_track_pool_mode_random_persisted(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--track-seeds", "0", "1",
            "--track-pool-mode", "random",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    assert cfg["track_pool_mode"] == "random"
    assert cfg["track_seeds"] == [0, 1]


def test_train_track_seeds_dim_mismatch(tmp_path):
    main = _make_main()

    class _FakePoolEnv:
        def __init__(self, track_seeds, track_kwargs=None, seed=None,
                     mode="round_robin", **env_kwargs):
            self.track_seeds = list(track_seeds)
            self.mode = mode
            self.envs = [
                types.SimpleNamespace(observation_dim=6),
                types.SimpleNamespace(observation_dim=8),
                types.SimpleNamespace(observation_dim=6),
            ]

    with patch("train.TrackPoolEnv", _FakePoolEnv), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        with pytest.raises(ValueError, match="observation_dim"):
            main([
                "--track-seeds", "0", "1", "2",
                "--episodes", "1",
                "--max-steps", "1",
                "--save-dir", str(tmp_path),
            ])


def test_train_single_seed_path_unchanged(tmp_path):
    main = _make_main()
    env_init_count = []
    real_init_env = RacingEnv.__init__

    def tracking_env(self, **kwargs):
        env_init_count.append(kwargs)
        real_init_env(self, **kwargs)

    with patch.object(RacingEnv, "__init__", tracking_env), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--track-seed", "5",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    assert cfg["track_seed"] == 5
    assert cfg["track_seeds"] is None
    assert cfg["track_pool_mode"] == "round_robin"
    assert len(env_init_count) == 1


# ---------------------------------------------------------------------------
# Gradient clipping (--max-grad-norm) tests
# ---------------------------------------------------------------------------

def test_train_max_grad_norm_persisted_to_config(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
            "--max-grad-norm", "1.0",
        ])
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["max_grad_norm"] == 1.0


def test_train_max_grad_norm_default_null_in_config(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert "max_grad_norm" in cfg
    assert cfg["max_grad_norm"] is None


def test_train_max_grad_norm_passed_to_agent(tmp_path):
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
            "--max-grad-norm", "1.0",
        ])

    kwargs = captured[0]
    assert kwargs["max_grad_norm"] == 1.0


def test_train_max_grad_norm_default_none_passed_to_agent(tmp_path):
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
    assert kwargs["max_grad_norm"] is None


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


def test_compare_policies_live_mode_skips_gif(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        import compare_policies
        model_path = str(tmp_path / "test_model.npz")
        _make_agent_checkpoint(model_path, state_dim=6)
        with patch.object(compare_policies, "_render_live_comparison") as live, \
             patch.object(compare_policies, "_save_comparison_gif") as save_gif:
            compare_policies.main([
                "--model-path", model_path,
                "--max-steps", "3",
                "--save-dir", str(tmp_path),
                "--live",
            ])
        live.assert_called_once()
        save_gif.assert_not_called()
    finally:
        sys.path[:] = orig_path


def _make_compare_policies_main():
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from compare_policies import main
        return main
    finally:
        sys.path[:] = orig_path


def _make_compare_policies_module():
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        import compare_policies as cp
        return cp
    finally:
        sys.path[:] = orig_path


def test_compare_policies_reads_config_from_model_dir(tmp_path):
    main = _make_compare_policies_main()
    model_path = str(tmp_path / "best_model.npz")
    _make_agent_checkpoint(model_path, state_dim=9)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"observation_mode": "local"}))
    main([
        "--model-path", model_path,
        "--save-dir", str(tmp_path),
        "--max-steps", "3",
    ])
    gif = tmp_path / "trained_vs_random.gif"
    assert gif.exists()
    assert gif.stat().st_size > 0


def test_compare_policies_explicit_config_local_mode(tmp_path):
    main = _make_compare_policies_main()
    model_path = str(tmp_path / "best_model.npz")
    _make_agent_checkpoint(model_path, state_dim=9)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"observation_mode": "local"}))
    main([
        "--model-path", model_path,
        "--config", str(cfg),
        "--save-dir", str(tmp_path),
        "--max-steps", "3",
    ])
    gif = tmp_path / "trained_vs_random.gif"
    assert gif.exists()
    assert gif.stat().st_size > 0


def test_compare_policies_state_dim_mismatch_raises(tmp_path):
    # 6-dim "state-mode" model against a "local-mode" config (obs_dim=9).
    main = _make_compare_policies_main()
    model_path = str(tmp_path / "best_model.npz")
    _make_agent_checkpoint(model_path, state_dim=6)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"observation_mode": "local"}))
    with pytest.raises(ValueError, match="observation_dim"):
        main([
            "--model-path", model_path,
            "--config", str(cfg),
            "--save-dir", str(tmp_path),
            "--max-steps", "3",
        ])


def test_compare_policies_random_actions_restricted_by_default(tmp_path):
    cp = _make_compare_policies_module()
    main = cp.main
    model_path = str(tmp_path / "best_model.npz")
    _make_agent_checkpoint(model_path, state_dim=6)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"allow_idle_actions": False}))

    emitted = []
    real_factory = cp._random_action_factory

    def recording_factory(rng, allow_idle_actions):
        inner = real_factory(rng, allow_idle_actions)

        def recorder(state):
            action_idx = inner(state)
            emitted.append(int(action_idx))
            return action_idx
        return recorder

    with patch.object(cp, "_random_action_factory", recording_factory):
        main([
            "--model-path", model_path,
            "--config", str(cfg),
            "--save-dir", str(tmp_path),
            "--max-steps", "8",
        ])
    assert emitted, "random policy never sampled an action"
    assert set(emitted).issubset({0, 1, 2})


def test_compare_policies_allow_idle_actions_flag_expands_actions(tmp_path):
    cp = _make_compare_policies_module()
    main = cp.main
    model_path = str(tmp_path / "best_model.npz")
    _make_agent_checkpoint(model_path, state_dim=6)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"allow_idle_actions": False}))

    emitted = []
    real_factory = cp._random_action_factory

    def recording_factory(rng, allow_idle_actions):
        inner = real_factory(rng, allow_idle_actions)

        def recorder(state):
            action_idx = inner(state)
            emitted.append(int(action_idx))
            return action_idx
        return recorder

    with patch.object(cp, "_random_action_factory", recording_factory):
        main([
            "--model-path", model_path,
            "--config", str(cfg),
            "--allow-idle-actions",
            "--save-dir", str(tmp_path),
            "--max-steps", "20",
        ])
    full_range = set(range(5))
    assert set(emitted).issubset(full_range)
    # When idle actions are restored, the random factory samples only via
    # rng.randint(len(ACTIONS)); seeing {3, 4} is impossible under the
    # restricted {0, 1, 2}-only branch, so its presence confirms the flag.
    assert (set(emitted) & {3, 4}), \
        "expected at least one idle action under --allow-idle-actions"


def test_compare_policies_track_seed_overrides_config(tmp_path):
    cp = _make_compare_policies_module()
    main = cp.main
    model_path = str(tmp_path / "best_model.npz")
    _make_agent_checkpoint(model_path, state_dim=6)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"track_seed": 0, "observation_mode": "state"}))

    captured_seeds = []
    real_make_track = cp._make_track

    def tracking_make_track(config):
        captured_seeds.append(config.get("track_seed"))
        return real_make_track(config)

    with patch.object(cp, "_make_track", tracking_make_track):
        main([
            "--model-path", model_path,
            "--config", str(cfg),
            "--track-seed", "5",
            "--save-dir", str(tmp_path),
            "--max-steps", "3",
        ])
    assert captured_seeds, "track was never built"
    assert all(s == 5 for s in captured_seeds)


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


# ---------------------------------------------------------------------------
# Render checkpoints script tests
# ---------------------------------------------------------------------------

def _make_render_checkpoints():
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        import render_checkpoints
        return render_checkpoints
    finally:
        sys.path[:] = orig_path


def _train_with_checkpoints(tmp_path, episodes=2, max_steps=1, freq=2):
    train_main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0):
        train_main([
            "--episodes", str(episodes),
            "--max-steps", str(max_steps),
            "--checkpoint-freq", str(freq),
            "--save-dir", str(tmp_path),
        ])


def test_render_checkpoints_generates_gif(tmp_path):
    mod = _make_render_checkpoints()
    _train_with_checkpoints(tmp_path, episodes=2, max_steps=1, freq=2)
    assert (tmp_path / "checkpoint_ep2.npz").exists()

    mod.main([
        "--checkpoint-dir", str(tmp_path),
        "--track-seed", "0",
        "--save-dir", str(tmp_path),
        "--max-steps", "2",
    ])
    gifs = list(tmp_path.glob("checkpoint_evolution*.gif"))
    assert len(gifs) == 1
    assert gifs[0].stat().st_size > 0


def test_render_checkpoints_mp4_flag_calls_video_export(tmp_path):
    mod = _make_render_checkpoints()
    _train_with_checkpoints(tmp_path, episodes=2, max_steps=1, freq=2)
    with patch("numpy_rl_racer.rendering.matplotlib_renderer."
               "MatplotlibRenderer.save_video") as save_video:
        mod.main([
            "--checkpoint-dir", str(tmp_path),
            "--track-seed", "0",
            "--save-dir", str(tmp_path),
            "--max-steps", "2",
            "--mp4",
            "--record-fps", "24",
        ])
    save_video.assert_called_once()
    assert save_video.call_args.kwargs["fps"] == 24


def test_render_checkpoints_explicit_order(tmp_path):
    mod = _make_render_checkpoints()
    # Create two real checkpoints with distinct episode numbers.
    _train_with_checkpoints(tmp_path, episodes=2, max_steps=1, freq=2)
    _train_with_checkpoints(tmp_path, episodes=4, max_steps=1, freq=4)
    ep2 = str(tmp_path / "checkpoint_ep2.npz")
    ep4 = str(tmp_path / "checkpoint_ep4.npz")

    load_order = []
    real_load = DQNAgent.load

    def tracking_load(self, path):
        load_order.append(int(os.path.basename(path).split("ep")[1].split(".")[0]))
        real_load(self, path)

    with patch.object(DQNAgent, "load", tracking_load), \
         patch.object(DQNAgent, "act", return_value=0):
        mod.main([
            "--checkpoints", ep4, ep2,
            "--track-seed", "0",
            "--save-dir", str(tmp_path),
            "--max-steps", "1",
        ])
    assert load_order == [4, 2]


def test_render_checkpoints_empty_dir_errors(tmp_path):
    mod = _make_render_checkpoints()
    with pytest.raises(ValueError, match="No checkpoint files found"):
        mod.main([
            "--checkpoint-dir", str(tmp_path),
            "--track-seed", "0",
            "--save-dir", str(tmp_path),
        ])


def test_render_checkpoints_sorted_by_episode(tmp_path):
    mod = _make_render_checkpoints()
    # Create checkpoint files with non-monotonic episode numbers; use state_dim=6
    # to match the default eval env (observation_mode="state", no obstacles).
    agent = DQNAgent(state_dim=6, hidden_sizes=[16], seed=0)
    for ep in (10, 2, 5):
        agent.save(str(tmp_path / f"checkpoint_ep{ep}.npz"))

    load_order = []
    real_load = DQNAgent.load

    def tracking_load(self, path):
        load_order.append(int(os.path.basename(path).split("ep")[1].split(".")[0]))
        real_load(self, path)

    with patch.object(DQNAgent, "load", tracking_load), \
         patch.object(DQNAgent, "act", return_value=0):
        mod.main([
            "--checkpoint-dir", str(tmp_path),
            "--track-seed", "0",
            "--save-dir", str(tmp_path),
            "--max-steps", "1",
        ])
    assert load_order == [2, 5, 10]


# ---------------------------------------------------------------------------
# NoisyNet (--noisy-net) tests
# ---------------------------------------------------------------------------


def test_train_noisy_net_flag_sets_use_noisy(tmp_path):
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
            "--noisy-net",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    assert captured[0]["use_noisy"] is True

    config_path = os.path.join(tmp_path, "config.json")
    assert os.path.exists(config_path)
    with open(config_path) as f:
        cfg = json.load(f)
    assert cfg["noisy_net"] is True


def test_train_default_no_noisy_net(tmp_path):
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

    assert captured[0]["use_noisy"] is False
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["noisy_net"] is False


def test_evaluate_loads_noisy_checkpoint(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from evaluate import _load_agent
        agent = DQNAgent(state_dim=6, hidden_sizes=[16], use_noisy=True, seed=42)
        model_path = str(tmp_path / "noisy_model.npz")
        agent.save(model_path)

        args = argparse.Namespace(model_path=model_path)
        loaded = _load_agent(args)
        assert loaded.use_noisy is True
    finally:
        sys.path[:] = orig_path


# ---------------------------------------------------------------------------
# Loss type (--loss-type / --huber-delta) tests
# ---------------------------------------------------------------------------

def _parse_loss_type_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-type", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    return parser.parse_args(args)


def test_train_loss_type_default_mse():
    parsed = _parse_loss_type_args([])
    assert parsed.loss_type == "mse"
    assert parsed.huber_delta == 1.0


def test_train_loss_type_huber_parsed():
    parsed = _parse_loss_type_args(["--loss-type", "huber", "--huber-delta", "1.0"])
    assert parsed.loss_type == "huber"
    assert parsed.huber_delta == 1.0


def test_train_loss_type_invalid_rejected():
    with pytest.raises(SystemExit):
        _parse_loss_type_args(["--loss-type", "rmse"])


def test_train_loss_type_huber_passed_to_agent(tmp_path):
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
            "--loss-type", "huber",
            "--huber-delta", "1.0",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    kwargs = captured[0]
    assert kwargs["loss_type"] == "huber"
    assert kwargs["huber_delta"] == 1.0


def test_train_loss_type_default_passed_to_agent(tmp_path):
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
    assert kwargs["loss_type"] == "mse"
    assert kwargs["huber_delta"] == 1.0


def test_train_loss_type_huber_runs(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--loss-type", "huber",
            "--huber-delta", "1.0",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    assert os.path.exists(config_path)
    with open(config_path) as f:
        cfg = json.load(f)
    assert cfg["loss_type"] == "huber"
    assert cfg["huber_delta"] == 1.0


# ---------------------------------------------------------------------------
# Soft target-network updates (--tau) tests
# ---------------------------------------------------------------------------

def _parse_tau_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.0)
    return parser.parse_args(args)


def test_train_tau_default_zero():
    parsed = _parse_tau_args([])
    assert parsed.tau == 0.0


def test_train_tau_parsed():
    parsed = _parse_tau_args(["--tau", "0.005"])
    assert parsed.tau == 0.005


def test_train_tau_default_zero_passed_to_agent(tmp_path):
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

    assert captured[0]["tau"] == 0.0


def test_train_tau_passed_to_agent(tmp_path):
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
            "--tau", "0.005",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])

    assert captured[0]["tau"] == 0.005


def test_train_tau_persisted_to_config(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--tau", "0.005",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["tau"] == 0.005


def test_train_tau_default_zero_in_config(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(tmp_path),
        ])
    with open(os.path.join(tmp_path, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["tau"] == 0.0


def test_train_tau_round_trip_via_config(tmp_path):
    main = _make_main()
    captured = []
    real_init = DQNAgent.__init__

    def tracking_init(self, **kwargs):
        captured.append(kwargs)
        real_init(self, **kwargs)

    save_dir = tmp_path / "first"
    save_dir.mkdir()
    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--tau", "0.005",
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(save_dir),
        ])

    assert captured[0]["tau"] == 0.005
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["tau"] == 0.005

    captured.clear()
    reloaded_dir = tmp_path / "second"
    with patch.object(DQNAgent, "__init__", tracking_init), \
         patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        main([
            "--config", os.path.join(save_dir, "config.json"),
            "--episodes", "1",
            "--max-steps", "1",
            "--save-dir", str(reloaded_dir),
        ])

    assert captured[0]["tau"] == 0.005


def test_train_tau_invalid_negative(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        with pytest.raises(ValueError, match="--tau must be in"):
            main([
                "--tau", "-0.1",
                "--episodes", "1",
                "--max-steps", "1",
                "--save-dir", str(tmp_path),
            ])


def test_train_tau_invalid_above_one(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0), \
         patch.object(DQNAgent, "save"):
        with pytest.raises(ValueError, match="--tau must be in"):
            main([
                "--tau", "1.5",
                "--episodes", "1",
                "--max-steps", "1",
                "--save-dir", str(tmp_path),
            ])


def test_train_tau_soft_update_runs(tmp_path):
    main = _make_main()
    with patch.object(DQNAgent, "act", return_value=0), \
         patch.object(DQNAgent, "train_step", return_value=0.0):
        main([
            "--tau", "0.005",
            "--episodes", "2",
            "--max-steps", "3",
            "--save-dir", str(tmp_path),
        ])
    config_path = os.path.join(tmp_path, "config.json")
    assert os.path.exists(config_path)
    with open(config_path) as f:
        cfg = json.load(f)
    assert cfg["tau"] == 0.005
    final_model = os.path.join(tmp_path, "final_model.npz")
    assert os.path.exists(final_model)
