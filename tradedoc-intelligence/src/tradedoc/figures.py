"""The one figure: confidence threshold against auto-approval rate."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .evaluate import RunOutput  # noqa: E402


def stp_curve(out: RunOutput, path: str, error_budget: float = 0.02) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    thr = [p.threshold for p in out.points]
    stp = [100 * p.stp_rate for p in out.points]
    err = [100 * p.error_rate for p in out.points]
    rec = [100 * p.recall for p in out.points]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    ax.plot(thr, stp, color="#1f4e79", lw=2.0, label="Auto-approval (STP) rate")
    ax.plot(thr, rec, color="#7f8c8d", lw=1.4, ls="--", label="Recall on correct docs")
    ax.set_xlabel("Confidence threshold (min over fields and line-item table)")
    ax.set_ylabel("Share of documents (%)")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, lw=0.6)

    ax2 = ax.twinx()
    ax2.plot(thr, err, color="#b03a2e", lw=1.6, label="Escaped-error rate (auto-approved)")
    ax2.axhline(100 * error_budget, color="#b03a2e", lw=0.9, ls=":", alpha=0.8)
    ax2.set_ylabel("Escaped-error rate among auto-approved (%)", color="#b03a2e")
    ax2.tick_params(axis="y", colors="#b03a2e")
    ax2.set_ylim(0, max(5.0, max(err) * 1.1))

    op: Optional = out.operating
    if op is not None:
        ax.axvline(op.threshold, color="#117a65", lw=1.0, ls="-.")
        ax.annotate(
            f"operating point t={op.threshold:.2f}\nSTP {100 * op.stp_rate:.1f}% at "
            f"{100 * op.error_rate:.1f}% error",
            xy=(op.threshold, 100 * op.stp_rate), xytext=(0.06, 24),
            fontsize=8, color="#117a65",
            arrowprops=dict(arrowstyle="->", color="#117a65", lw=0.8),
        )
    handles = ax.get_lines()[:2] + ax2.get_lines()[:1]
    ax.legend(handles, [h.get_label() for h in handles], loc="lower left", fontsize=8,
              frameon=False)
    ax.set_title("Straight-through processing against confidence threshold "
                 f"(n={len(out.confidences)} documents)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
