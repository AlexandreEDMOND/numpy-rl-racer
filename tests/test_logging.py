import csv
import math
import os

from numpy_rl_racer.utils.logging import TrainingLogger


def test_logger_creates_csv_with_headers(tmp_path):
    filepath = str(tmp_path / "log.csv")
    logger = TrainingLogger(filepath)
    logger.close()

    with open(filepath) as f:
        reader = csv.reader(f)
        headers = next(reader)

    assert headers == ["episode", "total_reward", "steps", "avg_loss", "epsilon", "avg_q_value"]


def test_logger_rows_contain_correct_types(tmp_path):
    filepath = str(tmp_path / "log.csv")
    logger = TrainingLogger(filepath)

    logger.log(episode=1, total_reward=10.5, steps=50, avg_loss=0.05, epsilon=0.9, avg_q_value=1.2)
    logger.log(episode=2, total_reward=-3.0, steps=100, avg_loss=float("nan"), epsilon=0.5, avg_q_value=float("nan"))
    logger.close()

    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert int(rows[0]["episode"]) == 1
    assert float(rows[0]["total_reward"]) == 10.5
    assert int(rows[0]["steps"]) == 50
    assert float(rows[0]["avg_loss"]) == 0.05
    assert float(rows[0]["epsilon"]) == 0.9
    assert float(rows[0]["avg_q_value"]) == 1.2

    assert math.isnan(float(rows[1]["avg_loss"]))
    assert math.isnan(float(rows[1]["avg_q_value"]))


def test_logger_multiple_episodes_sequential(tmp_path):
    filepath = str(tmp_path / "log.csv")
    logger = TrainingLogger(filepath)

    for ep in range(1, 6):
        logger.log(episode=ep, total_reward=float(ep * 10), steps=ep * 20, avg_loss=0.1 / ep, epsilon=max(0.1, 1.0 - ep * 0.2), avg_q_value=float(ep))
    logger.close()

    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5
    for i, row in enumerate(rows):
        expected_ep = i + 1
        assert int(row["episode"]) == expected_ep
        assert float(row["total_reward"]) == float(expected_ep * 10)
        assert int(row["steps"]) == expected_ep * 20


def test_logger_creates_nested_directory(tmp_path):
    nested = str(tmp_path / "a" / "b" / "c" / "log.csv")
    logger = TrainingLogger(nested)
    logger.log(episode=1, total_reward=0.0, steps=1, avg_loss=0.0, epsilon=1.0, avg_q_value=0.0)
    logger.close()

    assert os.path.exists(nested)
    with open(nested) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1


def test_logger_custom_fieldnames(tmp_path):
    filepath = str(tmp_path / "custom.csv")
    fieldnames = ["episode", "score"]
    logger = TrainingLogger(filepath, fieldnames=fieldnames)
    logger.log(episode=1, score=100.0)
    logger.close()

    with open(filepath) as f:
        reader = csv.reader(f)
        headers = next(reader)
    assert headers == fieldnames


def test_logger_close_idempotent(tmp_path):
    filepath = str(tmp_path / "log.csv")
    logger = TrainingLogger(filepath)
    logger.log(episode=1, total_reward=0.0, steps=1, avg_loss=0.0, epsilon=1.0, avg_q_value=0.0)

    logger.close()
    logger.close()  # should not raise

    with open(filepath) as f:
        rows = list(f)
    assert len(rows) == 2  # header + 1 data row


def test_logger_lr_column_when_fieldnames_include_lr(tmp_path):
    filepath = str(tmp_path / "log.csv")
    fieldnames = ["episode", "avg_loss", "lr"]
    logger = TrainingLogger(filepath, fieldnames=fieldnames)
    logger.log(episode=1, avg_loss=0.5, lr=0.01)
    logger.close()

    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert float(rows[0]["lr"]) == 0.01


def test_logger_data_preserved_after_close(tmp_path):
    filepath = str(tmp_path / "log.csv")
    logger = TrainingLogger(filepath)
    logger.log(episode=1, total_reward=5.0, steps=10, avg_loss=0.01, epsilon=0.8, avg_q_value=0.5)
    logger.close()

    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert float(rows[0]["total_reward"]) == 5.0
