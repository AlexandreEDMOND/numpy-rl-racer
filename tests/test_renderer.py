import matplotlib
matplotlib.use("Agg")

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
