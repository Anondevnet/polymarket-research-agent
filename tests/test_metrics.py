import math

import pytest

from pm_research.evals.metrics import (
    brier_score,
    calibration_bins,
    expected_calibration_error,
    log_loss,
    mrr,
    recall_at_k,
)


def test_brier() -> None:
    assert brier_score([1, 0], [1, 0]) == 0
    assert brier_score([0.5, 0.5], [1, 0]) == 0.25
    assert brier_score([0.8], [0]) == pytest.approx(0.64)


def test_log_loss() -> None:
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert log_loss([1.0], [0]) < 20  # clipped, finite


def test_calibration() -> None:
    probs = [0.1, 0.1, 0.9, 0.9]
    outs = [0, 0, 1, 1]
    bins = calibration_bins(probs, outs, n_bins=10)
    assert [b["count"] for b in bins] == [2, 2]
    assert expected_calibration_error(probs, outs) == pytest.approx(0.1)


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        brier_score([0.5], [1, 0])


def test_retrieval_metrics() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert recall_at_k(["a"], set(), 1) == 0.0
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert mrr(["x"], {"a"}) == 0.0
