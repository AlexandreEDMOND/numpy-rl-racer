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


def test_max_speed_upper_bound():
    car = KinematicCar(max_speed=8.0)
    state = CarState(velocity=7.0)
    new_state = car.step(state, steering=0.0, acceleration=10.0, dt=1.0)
    assert np.isclose(new_state.velocity, 8.0)
    assert new_state.velocity <= car.max_speed


def test_heading_normalization():
    car = KinematicCar()
    state = CarState(heading=3.0 * np.pi)
    new_state = car.step(state, steering=0.0, acceleration=0.0, dt=0.0)
    assert np.isclose(new_state.heading, np.pi, atol=1e-10)


def test_heading_in_range():
    car = KinematicCar()
    state = CarState(heading=0.0, velocity=1.0)
    new_state = car.step(state, steering=10.0, acceleration=0.0, dt=1.0)
    assert -np.pi - 1e-9 <= new_state.heading <= np.pi + 1e-9


def test_steering_changes_heading():
    car = KinematicCar()
    state = CarState(heading=0.0, velocity=5.0)
    new_state = car.step(state, steering=0.5, acceleration=0.0, dt=1.0)
    assert new_state.heading > 0.0
    assert np.isclose(new_state.heading, 0.5)


def test_diagonal_motion():
    car = KinematicCar()
    state = CarState(heading=np.pi / 4.0, velocity=1.0)
    new_state = car.step(state, steering=0.0, acceleration=0.0, dt=1.0)
    assert np.isclose(new_state.x, np.cos(np.pi / 4.0))
    assert np.isclose(new_state.y, np.sin(np.pi / 4.0))


def test_dt_scaling():
    car = KinematicCar()
    state = CarState(velocity=5.0)
    short = car.step(state, steering=0.0, acceleration=0.0, dt=0.5)
    long = car.step(state, steering=0.0, acceleration=0.0, dt=1.0)
    assert np.isclose(long.x, 2.0 * short.x)


def test_zero_dt_preserves_position():
    car = KinematicCar()
    state = CarState(x=2.0, y=-1.0, heading=0.5, velocity=4.0)
    new_state = car.step(state, steering=1.0, acceleration=3.0, dt=0.0)
    assert np.isclose(new_state.x, 2.0)
    assert np.isclose(new_state.y, -1.0)
    assert np.isclose(new_state.velocity, 4.0)
