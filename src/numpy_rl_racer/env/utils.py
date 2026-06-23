import numpy as np


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))
