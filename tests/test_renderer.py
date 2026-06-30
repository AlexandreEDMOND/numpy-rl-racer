import matplotlib
matplotlib.use("Agg")

import pytest

from numpy_rl_racer.env.car import CarState
from numpy_rl_racer.env.racing_env import ProceduralTrack
from numpy_rl_racer.rendering import MatplotlibRenderer


def _track():
    return ProceduralTrack(seed=0, radius=6.0, track_width=2.0)


def test_renderer_can_be_instantiated():
    renderer = MatplotlibRenderer(_track())
    assert renderer.fig is not None
    assert renderer.ax is not None
    renderer.close()


def test_renderer_renders_state():
    track = _track()
    x, y, heading = track.start_position
    renderer = MatplotlibRenderer(track)
    renderer.render(CarState(x=x, y=y, heading=heading, velocity=2.0), step=0, reward=0.1)
    renderer.close()


def test_headless_renderer_saves_figure(tmp_path):
    track = _track()
    x, y, heading = track.start_position
    renderer = MatplotlibRenderer(track, headless=True)
    renderer.render(CarState(x=x, y=y, heading=heading, velocity=2.0), step=0, reward=0.1)
    save_path = tmp_path / "test_procedural_output.png"
    renderer.fig.savefig(str(save_path), dpi=150)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    renderer.close()


def test_boundary_lines_drawn_as_line2d():
    track = _track()
    x, y, heading = track.start_position
    renderer = MatplotlibRenderer(track, headless=True)
    renderer.render(CarState(x=x, y=y, heading=heading, velocity=2.0))
    lines = renderer.ax.get_lines()
    assert len(lines) > 0
    assert "#666666" in {line.get_color() for line in lines}
    renderer.close()


def test_recording_saves_nonempty_gif(tmp_path):
    track = _track()
    x, y, heading = track.start_position
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=x, y=y, heading=heading, velocity=2.0)
    renderer.start_recording()
    renderer.render(state, step=0, reward=0.1)
    renderer.render(state, step=1, reward=0.2)
    gif_path = tmp_path / "test.gif"
    renderer.save_animation(str(gif_path), fps=10)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    renderer.close()


def test_recording_gif_expected_frame_count(tmp_path):
    track = _track()
    x, y, heading = track.start_position
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=x, y=y, heading=heading, velocity=2.0)
    renderer.start_recording()
    n_frames = 5
    for i in range(n_frames):
        renderer.render(state, step=i, reward=0.1)
    gif_path = tmp_path / "test.gif"
    renderer.save_animation(str(gif_path), fps=10)
    from PIL import Image
    with Image.open(str(gif_path)) as img:
        assert img.n_frames == n_frames
    renderer.close()


def test_save_animation_without_recording_raises_error(tmp_path):
    renderer = MatplotlibRenderer(_track(), headless=True)
    with pytest.raises(RuntimeError, match="No frames recorded"):
        renderer.save_animation(str(tmp_path / "empty.gif"))
    renderer.close()


def test_save_video_without_recording_raises_error(tmp_path):
    renderer = MatplotlibRenderer(_track(), headless=True)
    with pytest.raises(RuntimeError, match="No frames recorded"):
        renderer.save_video(str(tmp_path / "empty.mp4"))
    renderer.close()


def test_save_video_without_ffmpeg_raises_error(tmp_path):
    from unittest.mock import patch

    track = _track()
    x, y, heading = track.start_position
    renderer = MatplotlibRenderer(track, headless=True)
    renderer.start_recording()
    renderer.render(CarState(x=x, y=y, heading=heading, velocity=2.0))
    with patch("numpy_rl_racer.rendering.matplotlib_renderer.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="requires ffmpeg"):
            renderer.save_video(str(tmp_path / "test.mp4"))
    renderer.close()


def test_save_animation_after_stop_recording_raises_error(tmp_path):
    track = _track()
    x, y, heading = track.start_position
    renderer = MatplotlibRenderer(track, headless=True)
    renderer.start_recording()
    renderer.render(CarState(x=x, y=y, heading=heading, velocity=2.0))
    renderer.stop_recording(clear=True)
    with pytest.raises(RuntimeError, match="No frames recorded"):
        renderer.save_animation(str(tmp_path / "stopped.gif"))
    renderer.close()


def test_save_animation_without_pillow_raises_import_error(tmp_path):
    import sys
    from unittest.mock import patch

    track = _track()
    x, y, heading = track.start_position
    renderer = MatplotlibRenderer(track, headless=True)
    renderer.start_recording()
    renderer.render(CarState(x=x, y=y, heading=heading, velocity=2.0))

    with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
        with pytest.raises(ImportError, match="Pillow is required for GIF recording"):
            renderer.save_animation(str(tmp_path / "no_pillow.gif"))
    renderer.close()
