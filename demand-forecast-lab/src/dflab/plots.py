"""Figures for the README and for reading results at a glance.

Kept deliberately small and matplotlib-only. The charts that matter here are
comparative, so the design rules are: one encoding per figure, a colour-blind
safe categorical ramp, values printed on the marks rather than inferred from a
gridline, and no chart that would not survive being printed in black and white
by a planner.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .classify import ADI_CUT, CV2_CUT, QUADRANTS  # noqa: E402

__all__ = [
    "PALETTE",
    "plot_accuracy_by_quadrant",
    "plot_demand_quadrants",
    "plot_reconciliation_gain",
    "plot_example_series",
]

# Okabe-Ito derived; distinguishable in greyscale and for the common CVD types.
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#8C6D31",
    "#525252",
]

_STYLE = {
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "legend.frameon": False,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


def _save(fig, path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def plot_accuracy_by_quadrant(
    by_quadrant: dict[str, dict[str, dict[str, float]]],
    path,
    *,
    metric: str = "wape",
    methods: list[str] | None = None,
    counts: dict[str, int] | None = None,
    title: str | None = None,
) -> str:
    """Grouped bars: one panel per demand quadrant, one bar per method.

    This is the figure the repository exists to produce. A single overall
    accuracy bar chart would hide the whole finding.
    """
    _apply_style()
    methods = methods or list(by_quadrant)
    n_q = len(QUADRANTS)
    fig, axes = plt.subplots(1, n_q, figsize=(3.05 * n_q, 4.4), sharey=False)
    if n_q == 1:
        axes = [axes]

    for ax, q in zip(axes, QUADRANTS):
        vals = [by_quadrant.get(mth, {}).get(q, {}).get(metric, np.nan) for mth in methods]
        vals = [v if np.isfinite(v) else 0.0 for v in vals]
        order = np.argsort(vals)
        sorted_methods = [methods[i] for i in order]
        sorted_vals = [vals[i] for i in order]
        colors = [
            PALETTE[2] if i == 0 else PALETTE[7] if sorted_methods[i] in
            ("naive", "zero", "mean", "drift") or sorted_methods[i].startswith(("snaive", "ma["))
            else PALETTE[0]
            for i in range(len(sorted_vals))
        ]
        ypos = np.arange(len(sorted_vals))
        ax.barh(ypos, sorted_vals, color=colors, height=0.72)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sorted_methods, fontsize=7.5)
        ax.invert_yaxis()
        span = max(sorted_vals) if sorted_vals else 0.0
        if span <= 0:
            span = 1.0
        for i, v in enumerate(sorted_vals):
            ax.text(v + span * 0.02, i, f"{v:.3f}", va="center", fontsize=7)
        ax.set_xlim(0, span * 1.22)
        label = q if counts is None else f"{q}  (n={counts.get(q, 0)})"
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel(metric.upper())
        ax.grid(axis="y", visible=False)

    fig.suptitle(
        title or f"{metric.upper()} by method and demand quadrant "
        "(rolling-origin backtest; green = best in quadrant, grey = baseline)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_demand_quadrants(profiles, path, *, title: str | None = None) -> str:
    """ADI vs CV^2 scatter with the Syntetos-Boylan-Croston cut-points."""
    _apply_style()
    adi = np.array([p.adi for p in profiles], dtype=float)
    cv2 = np.array([p.cv2 for p in profiles], dtype=float)
    mean_d = np.array([p.mean_demand for p in profiles], dtype=float)
    quad = [p.quadrant for p in profiles]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colour = dict(zip(QUADRANTS, [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]))
    size = 18 + 130 * (mean_d / max(mean_d.max(), 1e-9)) ** 0.5
    for q in QUADRANTS:
        sel = [i for i, x in enumerate(quad) if x == q]
        if not sel:
            continue
        ax.scatter(
            adi[sel],
            cv2[sel],
            s=size[sel],
            c=colour[q],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.6,
            label=f"{q} (n={len(sel)})",
        )
    ax.axvline(ADI_CUT, color="#333333", lw=1.0, ls="--")
    ax.axhline(CV2_CUT, color="#333333", lw=1.0, ls="--")
    ax.text(ADI_CUT, ax.get_ylim()[1], f" ADI={ADI_CUT}", va="top", fontsize=7.5)
    ax.text(ax.get_xlim()[1], CV2_CUT, f"CV²={CV2_CUT} ", ha="right", va="bottom", fontsize=7.5)
    ax.set_xlabel("ADI  (average demand interval, periods per demand)")
    ax.set_ylabel("CV²  (squared CV of non-zero demand sizes)")
    ax.set_title(
        title or "Demand classification of the synthetic panel (marker area ~ mean demand)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_reconciliation_gain(
    table: dict[str, dict[str, float]], path, *, title: str | None = None
) -> str:
    """Grouped bars of WAPE by reconciliation method and hierarchy level."""
    _apply_style()
    methods = list(table)
    levels = list(next(iter(table.values())))
    x = np.arange(len(levels))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(1.7 * len(levels) + 2.0, 4.2))
    for k, mth in enumerate(methods):
        vals = [table[mth].get(lv, np.nan) for lv in levels]
        ax.bar(
            x + k * width - 0.4 + width / 2,
            vals,
            width=width * 0.92,
            label=mth,
            color=PALETTE[k % len(PALETTE)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("WAPE")
    ax.set_title(title or "Reconciliation: WAPE by hierarchy level", fontsize=10)
    ax.legend(fontsize=8, ncol=min(len(methods), 4))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_example_series(
    panel, indices, labels, path, *, cut: int | None = None, title: str | None = None
) -> str:
    """Small multiples of representative series, one per quadrant."""
    _apply_style()
    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, i, lab in zip(axes, indices, labels):
        y = np.asarray(panel)[i]
        ax.plot(np.arange(y.size), y, lw=0.9, color=PALETTE[0])
        ax.fill_between(np.arange(y.size), 0, y, color=PALETTE[0], alpha=0.18)
        if cut is not None:
            ax.axvline(cut, color=PALETTE[3], lw=1.1, ls="--")
        ax.set_ylabel("units", fontsize=8)
        ax.set_title(lab, fontsize=9, loc="left")
    axes[-1].set_xlabel("week index")
    fig.suptitle(title or "Representative series from each demand quadrant", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)
