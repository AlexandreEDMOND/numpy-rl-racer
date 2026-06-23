import numpy as np

from numpy_rl_racer.env.utils import normalize_angle


def test_angles_within_range_unchanged():
    angles = [0.0, np.pi, -np.pi, np.pi / 2, -np.pi / 2, 1.0, -2.0]
    for a in angles:
        result = normalize_angle(a)
        assert np.isclose(result, a, atol=1e-15), f"Failed for {a}: got {result}"


def test_angles_outside_range():
    cases = [
        (3.0 * np.pi, np.pi),
        (-3.0 * np.pi, -np.pi),
        (5.0 * np.pi, np.pi),
        (-5.0 * np.pi, -np.pi),
        (1000.0, 1000.0 % (2 * np.pi) - (2 * np.pi if (1000.0 % (2 * np.pi)) > np.pi else 0)),
    ]
    for inp, expected in cases:
        result = normalize_angle(inp)
        assert np.isclose(result, expected, atol=1e-10), (
            f"normalize_angle({inp}) = {result}, expected {expected}"
        )


def test_large_angle():
    result = normalize_angle(1000.0)
    assert -np.pi <= result <= np.pi


def test_array_input():
    angles = np.array([0.0, np.pi, 3.0 * np.pi, -3.0 * np.pi])
    result = normalize_angle(angles)
    expected = np.array([0.0, np.pi, np.pi, -np.pi])
    np.testing.assert_array_almost_equal(result, expected, decimal=10)


def test_scalar_float():
    result = normalize_angle(3.0 * np.pi)
    assert isinstance(result, (float, np.floating))
    assert np.isclose(result, np.pi)


def test_result_in_neg_pi_to_pi():
    for a in np.linspace(-10.0 * np.pi, 10.0 * np.pi, 100):
        result = normalize_angle(a)
        assert -np.pi - 1e-12 <= result <= np.pi + 1e-12, (
            f"normalize_angle({a}) = {result} outside [-π, π]"
        )


def test_regression_both_implementations():
    def old_impl_1(a):
        return np.arctan2(np.sin(a), np.cos(a))
    test_angles = np.linspace(-4 * np.pi, 4 * np.pi, 200)
    for a in test_angles:
        expected = old_impl_1(a)
        result = normalize_angle(a)
        assert np.isclose(result, expected, atol=1e-15), (
            f"Mismatch at {a}: old={expected}, new={result}"
        )
