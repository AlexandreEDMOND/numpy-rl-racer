import argparse
import os

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
