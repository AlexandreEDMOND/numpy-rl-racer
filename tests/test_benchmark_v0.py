import os
import sys


def _make_benchmark():
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    orig_path = sys.path.copy()
    sys.path.insert(0, scripts_dir)
    try:
        from benchmark_v0 import BenchmarkConfig, run_benchmark
        return BenchmarkConfig, run_benchmark
    finally:
        sys.path[:] = orig_path


def test_v0_benchmark_smoke_distance(tmp_path):
    BenchmarkConfig, run_benchmark = _make_benchmark()

    result = run_benchmark(
        BenchmarkConfig(
            episodes=10,
            max_steps=80,
            eval_freq=5,
            eval_episodes=2,
            seed=0,
            save_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
            batch_size=16,
            buffer_size=1000,
            epsilon_decay=0.99,
            smoke_steps=80,
            smoke_distance_threshold=2.0,
        )
    )

    assert result.best_eval_reward > -float("inf")
    assert result.smoke_passed
    assert result.smoke_distance > 2.0
