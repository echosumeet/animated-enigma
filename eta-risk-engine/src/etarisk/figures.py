"""The one figure: reliability of the delay-risk model, raw vs isotonic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["calibration_figure"]


def calibration_figure(metrics: dict, outdir: str | Path = "docs") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = metrics["reliability_raw"]
    cal = metrics["reliability_calibrated"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="0.55", lw=1.0, ls="--", label="perfect calibration")
    ax.plot(raw["predicted"], raw["observed"], "o-", color="#b4453c", lw=1.6, ms=5,
            label=f"raw GBT (ECE {metrics['ece_raw']:.3f})")
    ax.plot(cal["predicted"], cal["observed"], "s-", color="#2f6f8f", lw=1.6, ms=5,
            label=f"isotonic (ECE {metrics['ece_calibrated']:.3f})")
    thr = metrics["cost_thresholds"]["notify_to_expedite"]
    ax.axvline(thr, color="#4a7a3a", lw=1.0, ls=":")
    ax.text(thr + 0.012, 0.05, "expedite threshold", fontsize=8, color="#4a7a3a", rotation=90)
    ax.set_xlabel("predicted P(late)")
    ax.set_ylabel("observed late rate")
    ax.set_title("Reliability on the held-out test block")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(metrics["p_test_calibrated"], bins=30, color="#2f6f8f", alpha=0.8)
    ax.axvline(thr, color="#4a7a3a", lw=1.0, ls=":")
    ax.set_xlabel("calibrated P(late)")
    ax.set_ylabel("shipments")
    ax.set_title("Risk distribution")
    ax.grid(alpha=0.25)

    fig.suptitle("Delay-risk calibration, isotonic on a temporally later block", fontsize=11)
    fig.tight_layout()
    path = outdir / "calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
