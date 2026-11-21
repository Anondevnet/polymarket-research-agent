from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import calibration_bins


def calibration_plot(
    forecasts: Mapping[str, Sequence[float]],
    outcomes: Sequence[int],
    path: str | Path,
    n_bins: int = 5,
    title: str = "Calibration (reliability diagram)",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    for name, probs in forecasts.items():
        bins = calibration_bins(probs, outcomes, n_bins)
        ax.plot(
            [b["mean_pred"] for b in bins],
            [b["frac_pos"] for b in bins],
            marker="o",
            label=f"{name} (n={len(probs)})",
        )
    ax.set_xlabel("Predicted P(YES)")
    ax.set_ylabel("Observed frequency of YES")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
