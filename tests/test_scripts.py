import argparse

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
