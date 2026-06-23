import numpy_rl_racer


def test_package_imports():
    assert numpy_rl_racer is not None


def test_package_version():
    assert isinstance(numpy_rl_racer.__version__, str)
    assert len(numpy_rl_racer.__version__) > 0
