import numpy as np
from numpy_rl_racer.env import CarState, KinematicCar


def test_forward_motion():
    car = KinematicCar()
    state = CarState(x=0.0, y=0.0, heading=0.0, velocity=5.0)
    new_state = car.step(state, steering=0.0, acceleration=0.0, dt=1.0)
    assert np.isclose(new_state.x, 5.0)
    assert np.isclose(new_state.y, 0.0)
    assert np.isclose(new_state.heading, 0.0)
    assert np.isclose(new_state.velocity, 5.0)


def test_acceleration():
    car = KinematicCar()
    state = CarState(velocity=0.0)
    new_state = car.step(state, steering=0.0, acceleration=5.0, dt=1.0)
    assert np.isclose(new_state.velocity, 5.0)


def test_braking_velocity_lower_bound():
    car = KinematicCar()
    state = CarState(velocity=3.0)
    new_state = car.step(state, steering=0.0, acceleration=-10.0, dt=1.0)
    assert np.isclose(new_state.velocity, 0.0)
    assert new_state.velocity >= 0.0


def test_heading_normalization():
    car = KinematicCar()
    state = CarState(heading=3.0 * np.pi)
    new_state = car.step(state, steering=0.0, acceleration=0.0, dt=0.0)
    assert np.isclose(new_state.heading, np.pi, atol=1e-10)
