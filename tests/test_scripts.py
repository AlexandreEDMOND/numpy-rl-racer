import argparse
import csv
import os
import sys
from unittest.mock import patch

import numpy as np

from numpy_rl_racer.agent.dqn import DQNAgent
from numpy_rl_racer.env import CircularTrack, RacingEnv, RectangularTrack


def _parse_track(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["rectangular", "circular"], default="rectangular")
    parsed = parser.parse_args(args)
    if parsed.track == "circular":
        track = CircularTrack(radius=6.0, track_width=2.0)
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


def test_evaluate_headless(tmp_path):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from evaluate import main
        with patch("numpy_rl_racer.agent.dqn.DQNAgent.load"):
            main([
                "--headless",
                "--episodes", "1",
                "--max-steps", "3",
                "--save-dir", str(tmp_path),
            ])
    finally:
        sys.path[:] = orig_path
    saved = list(tmp_path.glob("eval_ep*_final.png"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0


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
