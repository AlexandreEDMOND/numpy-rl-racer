import matplotlib
matplotlib.use("Agg")

from numpy_rl_racer.rendering import MatplotlibRenderer
from numpy_rl_racer.env.racing_env import RectangularTrack
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
