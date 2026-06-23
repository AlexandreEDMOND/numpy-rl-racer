import numpy as np


class CarState:
    def __init__(self, x=0.0, y=0.0, heading=0.0, velocity=0.0):
        self.x = np.float64(x)
        self.y = np.float64(y)
        self.heading = np.float64(heading)
        self.velocity = np.float64(velocity)

    def __repr__(self):
        return (
            f"CarState(x={self.x}, y={self.y}, "
            f"heading={self.heading}, velocity={self.velocity})"
        )


class KinematicCar:
    def __init__(self, max_speed=10.0):
        self.max_speed = np.float64(max_speed)

    def step(self, state, steering, acceleration, dt=0.1):
        heading = state.heading + np.float64(steering) * np.float64(dt)
        heading = _normalize_angle(heading)

        velocity = state.velocity + np.float64(acceleration) * np.float64(dt)
        velocity = np.clip(velocity, 0.0, self.max_speed)

        x = state.x + velocity * np.cos(heading) * np.float64(dt)
        y = state.y + velocity * np.sin(heading) * np.float64(dt)

        return CarState(x, y, heading, velocity)


def _normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))
