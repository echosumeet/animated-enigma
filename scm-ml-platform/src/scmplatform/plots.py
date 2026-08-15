"""The single figure this repository ships: feature drift against a fixed reference."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .monitoring import PSI_MAJOR, PSI_MODERATE  # noqa: E402


def plot_drift_over_time(drift: "object", path: str | Path, title: str = "Feature drift vs fixed reference window"):
    """Plot PSI per feature per window from :func:`scmplatform.monitoring.drift_over_time`."""
    path = Path(path)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for feature, grp in drift.groupby("feature"):
        grp = grp.sort_values("window_start")
        ax.plot(grp["window_start"], grp["psi"], marker="o", markersize=4, linewidth=1.6, label=feature)
    ax.axhline(PSI_MODERATE, color="#B8860B", linestyle="--", linewidth=1.0)
    ax.axhline(PSI_MAJOR, color="#B22222", linestyle="--", linewidth=1.0)
    ax.text(0.005, PSI_MODERATE, " investigate (PSI 0.10)", color="#B8860B", va="bottom", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.text(0.005, PSI_MAJOR, " page on-call (PSI 0.25)", color="#B22222", va="bottom", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("Monitoring window start")
    ax.set_ylabel("Population Stability Index")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
