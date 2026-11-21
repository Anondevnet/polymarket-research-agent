"""Forecast and retrieval metrics (pure numpy, unit-tested)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _arrays(probs: Sequence[float], outcomes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probs and outcomes must be non-empty and the same length")
    return p, y


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of probabilities vs 0/1 outcomes. 0 is perfect, 0.25 = always 50%."""
    p, y = _arrays(probs, outcomes)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: Sequence[float], outcomes: Sequence[int], eps: float = 1e-6) -> float:
    """Mean negative log-likelihood (natural log). ln 2 = 0.693 for always 50%."""
    p, y = _arrays(probs, outcomes)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_bins(
    probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> list[dict[str, float]]:
    """Reliability-diagram bins: mean predicted prob vs observed frequency per bin."""
    p, y = _arrays(probs, outcomes)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            bins.append(
                {
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "count": int(mask.sum()),
                    "mean_pred": float(p[mask].mean()),
                    "frac_pos": float(y[mask].mean()),
                }
            )
    return bins


def expected_calibration_error(
    probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> float:
    p, _ = _arrays(probs, outcomes)
    bins = calibration_bins(probs, outcomes, n_bins)
    return float(sum(b["count"] * abs(b["mean_pred"] - b["frac_pos"]) for b in bins) / p.size)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / rank
    return 0.0
