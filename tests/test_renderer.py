import matplotlib
matplotlib.use("Agg")

import pytest
from unittest.mock import patch

from numpy_rl_racer.rendering import MatplotlibRenderer
from numpy_rl_racer.env.racing_env import CircularTrack, Figure8Track, RectangularTrack
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


# ── Figure8Track Renderer Tests ──────────────────────────────────────


def test_renderer_can_be_instantiated_with_figure8_track():
    track = Figure8Track(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track)
    assert renderer is not None
    assert renderer.fig is not None
    assert renderer.ax is not None
    renderer.close()


def test_renderer_renders_state_with_figure8_track():
    track = Figure8Track(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track)
    gx, gy = track.goal_position
    state = CarState(x=float(gx), y=float(gy), heading=0.0, velocity=2.0)
    renderer.render(state, step=0, reward=0.1)
    renderer.close()


def test_boundary_lines_drawn_as_line2d_figure8():
    track = Figure8Track(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track, headless=True)
    gx, gy = track.goal_position
    state = CarState(x=float(gx), y=float(gy), heading=0.0, velocity=2.0)
    renderer.render(state)
    lines = renderer.ax.get_lines()
    assert len(lines) > 0
    assert "#666666" in {line.get_color() for line in lines}
    renderer.close()


def test_headless_figure8_renderer_saves_figure(tmp_path):
    track = Figure8Track(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track, headless=True)
    gx, gy = track.goal_position
    state = CarState(x=float(gx), y=float(gy), heading=0.0, velocity=2.0)
    renderer.render(state, step=0, reward=0.1)
    renderer.render(state, step=1, reward=0.2)
    save_path = tmp_path / "test_figure8_output.png"
    renderer.fig.savefig(str(save_path), dpi=150)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    renderer.close()


def test_recording_figure8_saves_nonempty_gif(tmp_path):
    track = Figure8Track(radius=6.0, track_width=2.0)
    renderer = MatplotlibRenderer(track, headless=True)
    gx, gy = track.goal_position
    state = CarState(x=float(gx), y=float(gy), heading=0.0, velocity=2.0)
    renderer.start_recording()
    renderer.render(state, step=0, reward=0.1)
    renderer.render(state, step=1, reward=0.2)
    gif_path = tmp_path / "test_figure8.gif"
    renderer.save_animation(str(gif_path), fps=10)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    renderer.close()


# ── Reward Line Rendering Tests ───────────────────────────────────────


def test_renderer_accepts_reward_line_progress():
    track = RectangularTrack()
    reward_progress = [0.25, 0.5, 0.75]
    renderer = MatplotlibRenderer(track, headless=True, reward_line_progress=reward_progress)
    assert len(renderer._reward_line_endpoints) == 3
    renderer.close()


def test_reward_lines_drawn_when_provided():
    track = RectangularTrack()
    reward_progress = [0.25, 0.5, 0.75]
    renderer = MatplotlibRenderer(track, headless=True, reward_line_progress=reward_progress)
    state = CarState(x=0.0, y=0.0, heading=0.0, velocity=0.0)
    renderer.render(state)
    lines = renderer.ax.get_lines()
    # Should contain the reward lines with #2e86c1 color
    assert any("#2e86c1" in str(line.get_color()) for line in lines)
    renderer.close()


def test_reward_lines_empty_when_not_provided():
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    assert len(renderer._reward_line_endpoints) == 0
    renderer.close()


# ── Configurable DPI and FPS Tests ────────────────────────────────────


def test_renderer_custom_dpi():
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True, dpi=200)
    assert renderer.dpi == 200
    renderer.close()


def test_custom_dpi_produces_larger_frames(tmp_path):
    track = RectangularTrack()
    renderer_low = MatplotlibRenderer(track, headless=True, dpi=50)
    renderer_high = MatplotlibRenderer(track, headless=True, dpi=200)
    state = CarState(x=0.0, y=0.0, heading=0.0, velocity=0.0)
    renderer_low.start_recording()
    renderer_high.start_recording()
    renderer_low.render(state)
    renderer_high.render(state)
    low_frame = renderer_low._recording_frames[0]
    high_frame = renderer_high._recording_frames[0]
    assert low_frame.shape[0] < high_frame.shape[0]
    assert low_frame.shape[1] < high_frame.shape[1]
    # At dpi=50 on figsize=(8,6): 400x300 pixels; at dpi=200: 1600x1200
    assert low_frame.shape == (300, 400, 3)
    assert high_frame.shape == (1200, 1600, 3)
    renderer_low.close()
    renderer_high.close()


def test_renderer_stores_fps():
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True, fps=20)
    assert renderer._fps == 20
    renderer.close()


def test_save_animation_uses_renderer_fps(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True, fps=15)
    state = CarState(x=0.0, y=0.0, heading=0.0, velocity=0.0)
    renderer.start_recording()
    renderer.render(state)
    renderer.render(state)
    gif_path = tmp_path / "fps_test.gif"
    renderer.save_animation(str(gif_path))
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    renderer.close()


def test_save_animation_explicit_fps_overrides_renderer(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True, fps=5)
    state = CarState(x=0.0, y=0.0, heading=0.0, velocity=0.0)
    renderer.start_recording()
    renderer.render(state)
    renderer.render(state)
    gif_path = tmp_path / "override_fps.gif"
    renderer.save_animation(str(gif_path), fps=20)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    renderer.close()


def test_save_video_no_frames_raises_error(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    video_path = tmp_path / "empty.mp4"
    with pytest.raises(RuntimeError, match="No frames recorded"):
        renderer.save_video(str(video_path))
    renderer.close()


def test_save_video_ffmpeg_not_available(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    state = CarState(x=0.0, y=0.0, heading=0.0, velocity=0.0)
    renderer.start_recording()
    renderer.render(state)
    video_path = tmp_path / "no_ffmpeg.mp4"
    import subprocess
    original_run = subprocess.run
    def mock_run(*args, **kwargs):
        if args[0] and args[0][0] == "ffmpeg":
            raise subprocess.CalledProcessError(1, ["ffmpeg", "-version"])
        return original_run(*args, **kwargs)
    with patch("subprocess.run", mock_run):
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            renderer.save_video(str(video_path))
    renderer.close()


def test_gif_with_custom_dpi_and_fps(tmp_path):
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True, dpi=150, fps=20)
    state = CarState(x=0.0, y=0.0, heading=0.0, velocity=0.0)
    renderer.start_recording()
    n_frames = 5
    for i in range(n_frames):
        renderer.render(state, step=i, reward=0.1)
    gif_path = tmp_path / "custom.gif"
    renderer.save_animation(str(gif_path))
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    from PIL import Image
    with Image.open(str(gif_path)) as img:
        assert img.n_frames == n_frames
        w, h = img.size
        expected_w = int(8 * 150)
        expected_h = int(6 * 150)
        assert abs(w - expected_w) <= 2
        assert abs(h - expected_h) <= 2
    renderer.close()


def test_figure_proportions_consistent_across_dpi(tmp_path):
    track = RectangularTrack()
    state = CarState(x=0.0, y=0.0, heading=0.0, velocity=0.0)
    renderer1 = MatplotlibRenderer(track, headless=True, dpi=100)
    renderer1.start_recording()
    renderer1.render(state)
    frame_100 = renderer1._recording_frames[0]
    renderer1.close()
    renderer2 = MatplotlibRenderer(track, headless=True, dpi=200)
    renderer2.start_recording()
    renderer2.render(state)
    frame_200 = renderer2._recording_frames[0]
    renderer2.close()
    # Aspect ratio should be 8/6 = 1.333 regardless of DPI
    aspect_100 = frame_100.shape[1] / frame_100.shape[0]
    aspect_200 = frame_200.shape[1] / frame_200.shape[0]
    assert abs(aspect_100 - aspect_200) < 0.01


def test_renderer_backward_compatible_no_dpi_fps():
    """Calling without dpi/fps args should use defaults (dpi=100, fps=10)."""
    track = RectangularTrack()
    renderer = MatplotlibRenderer(track, headless=True)
    assert renderer.dpi == 100
    assert renderer._fps == 10
    renderer.close()
