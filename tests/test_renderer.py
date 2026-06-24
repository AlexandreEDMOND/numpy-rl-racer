import matplotlib
matplotlib.use("Agg")

import pytest

from numpy_rl_racer.rendering import MatplotlibRenderer
from numpy_rl_racer.env.racing_env import CircularTrack, RectangularTrack
from numpy_rl_racer.env.car import CarState


def test_renderer_can_be_instantiated():
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track)
    assert renderer is not None
    assert renderer.fig is not None
    assert renderer.ax is not None
    renderer.close()


def test_renderer_renders_state():
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track)
    state = CarState(x=1.0, y=1.0, heading=0.5, velocity=2.0)
    renderer.render(state, step=0, reward=0.1)
    renderer.close()


def test_renderer_can_be_instantiated_with_circular_track():
    track = CircularTrack(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track)
    assert renderer is not None
    assert renderer.fig is not None
    assert renderer.ax is not None
    renderer.close()


def test_renderer_renders_state_with_circular_track():
    track = CircularTrack(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track)
    state = CarState(x=0.0, y=-6.0, heading=0.0, velocity=2.0)
    renderer.render(state, step=0, reward=0.1)
    renderer.close()


def test_headless_circular_renderer_saves_figure(tmp_path):
    track = CircularTrack(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=0.0, y=-6.0, heading=0.0, velocity=2.0)
    renderer.render(state, step=0, reward=0.1)
    renderer.render(state, step=1, reward=0.2)
    save_path = tmp_path / "test_circular_output.png"
    renderer.fig.savefig(str(save_path), dpi=150)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    renderer.close()


def test_boundary_lines_drawn_as_line2d_rectangular():
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=1.0, y=1.0, heading=0.5, velocity=2.0)
    renderer.render(state)
    lines = renderer.ax.get_lines()
    assert len(lines) > 0
    assert "#666666" in {line.get_color() for line in lines}
    renderer.close()


def test_boundary_lines_drawn_as_line2d_circular():
    track = CircularTrack(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=0.0, y=-6.0, heading=0.0, velocity=2.0)
    renderer.render(state)
    lines = renderer.ax.get_lines()
    assert len(lines) > 0
    assert "#666666" in {line.get_color() for line in lines}
    renderer.close()


def test_recording_saves_nonempty_gif(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=1.0, y=1.0, heading=0.5, velocity=2.0)
    renderer.start_recording()
    renderer.render(state, step=0, reward=0.1)
    renderer.render(state, step=1, reward=0.2)
    gif_path = tmp_path / "test.gif"
    renderer.save_animation(str(gif_path), fps=10)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    renderer.close()


def test_recording_gif_expected_frame_count(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=1.0, y=1.0, heading=0.5, velocity=2.0)
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


def test_recording_headless_works(tmp_path):
    track = CircularTrack(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=0.0, y=-6.0, heading=0.0, velocity=2.0)
    renderer.start_recording()
    renderer.render(state)
    gif_path = tmp_path / "headless_test.gif"
    renderer.save_animation(str(gif_path), fps=10)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    renderer.close()


def test_save_animation_without_recording_raises_error(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    gif_path = tmp_path / "empty.gif"
    with pytest.raises(RuntimeError, match="No frames recorded"):
        renderer.save_animation(str(gif_path))
    renderer.close()


def test_save_animation_after_stop_recording_raises_error(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=1.0, y=1.0, heading=0.5, velocity=2.0)
    renderer.start_recording()
    renderer.render(state)
    renderer.stop_recording(clear=True)
    gif_path = tmp_path / "stopped.gif"
    with pytest.raises(RuntimeError, match="No frames recorded"):
        renderer.save_animation(str(gif_path))
    renderer.close()


def test_headless_renderer_saves_figure(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=1.0, y=1.0, heading=0.5, velocity=2.0)
    renderer.render(state, step=0, reward=0.1)
    renderer.render(state, step=1, reward=0.2)
    save_path = tmp_path / "test_output.png"
    renderer.fig.savefig(str(save_path), dpi=150)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    renderer.close()


def test_save_animation_without_pillow_raises_import_error(tmp_path):
    import sys
    from unittest.mock import patch

    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=1.0, y=1.0, heading=0.5, velocity=2.0)
    renderer.start_recording()
    renderer.render(state)
    gif_path = tmp_path / "no_pillow.gif"

    with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
        with pytest.raises(ImportError, match="Pillow is required for GIF recording"):
            renderer.save_animation(str(gif_path))
    renderer.close()
