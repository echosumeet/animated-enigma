#!/usr/bin/env python3
"""Regenerate the figures in ``docs/``.

Separate from ``run_benchmarks.py`` because matplotlib is an optional extra: the
library, the tests and the benchmark tables all run without it.

Palette notes: three categorical hues in fixed order (blue, orange, aqua),
validated for colorblind separation as a set; an ordinal blue ramp where the
dimension is ordered rather than nominal; a neutral gray reserved for reference
lines so a baseline is never mistaken for a series.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from echelonsim.bullwhip import measure_by_echelon  # noqa: E402
from echelonsim.disruption import run_disruption_study  # noqa: E402
from echelonsim.experiments import (  # noqa: E402
    DEFAULT_CONFIG,
    estimate_warmup,
    merge_config,
    run_scenario,
    system_inventory_path,
)
from echelonsim.experiments import _replication  # noqa: E402
from echelonsim.metrics import mser5_truncation  # noqa: E402
from echelonsim.tradeoffs import lead_time_grid  # noqa: E402

DOCS = os.path.join(REPO_ROOT, "docs")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8c8b86"
GRID = "#e6e5e2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order, never cycled
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]  # one hue, light -> dark
NEUTRAL = "#9a9992"

SEED = 20260215
BASE: Dict[str, Any] = merge_config(DEFAULT_CONFIG, {
    "topology": {"kind": "serial", "levels": 3},
    "forecast": {"kind": "exponential", "alpha": 0.3},
    "run": {"periods": 720, "replications": 30, "seed": SEED},
})


def style_axis(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)


def new_figure(width: float, height: float):
    figure = plt.figure(figsize=(width, height), dpi=160, facecolor=SURFACE)
    return figure


def save(figure, name: str) -> str:
    path = os.path.join(DOCS, name)
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------

def figure_bullwhip(warmup: int) -> None:
    """Order streams by echelon, and amplification against chain depth."""
    outcome = run_scenario(BASE, name="bullwhip", warmup=warmup, keep_results=True)
    result = outcome.results[0]
    window = slice(120, 200)
    periods = np.arange(window.start, window.stop)

    figure = new_figure(11.5, 5.0)
    grid = figure.add_gridspec(3, 2, width_ratios=[1.5, 1.0], wspace=0.22,
                               hspace=0.28)

    # Small multiples rather than one overplotted panel: three spiky series on
    # one axis hides the very thing the figure exists to show.
    names = [series.name for series in result.stocking_series()]
    demand = result.customer_demand[window]
    ceiling = max(
        result.nodes[name].orders_placed[window].max() for name in names
    )
    axes = []
    for index, name in enumerate(names):
        ax = figure.add_subplot(grid[index, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        style_axis(ax, "", "", "")
        ax.plot(periods, demand, color=NEUTRAL, linewidth=1.2, zorder=2)
        ax.plot(periods, result.nodes[name].orders_placed[window],
                color=SERIES[index], linewidth=1.8, zorder=3)
        ax.set_ylim(0, ceiling * 1.08)
        ax.text(0.012, 0.86, f"{name} orders", transform=ax.transAxes,
                color=SERIES[index], fontsize=10, fontweight="bold", va="top")
        if index == 0:
            ax.set_title("Order streams amplify at every echelon",
                         color=INK, fontsize=12, loc="left", pad=10)
            ax.text(0.012, 0.62, "grey line: end-customer demand",
                    transform=ax.transAxes, color=INK_MUTED, fontsize=8.5, va="top")
        if index < len(names) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("period", color=INK_SOFT, fontsize=10)
    axes[1].set_ylabel("units ordered", color=INK_SOFT, fontsize=10)

    ax2 = figure.add_subplot(grid[:, 1])
    style_axis(ax2, "stocking echelons in the chain",
               "Var(orders) / Var(end demand)",
               "Amplification compounds with depth")
    depths, values = [], []
    for levels in (1, 2, 3, 4):
        config = merge_config(BASE, {"topology": {"levels": levels}})
        run = run_scenario(config, name=f"levels={levels}", warmup=warmup,
                           keep_results=True)
        top = max(run.results[0].stocking_series(), key=lambda s: s.level)
        depths.append(levels)
        values.append(run.ratio_ci(f"var_orders:{top.name}", "var_demand").mean)
    bars = ax2.bar(depths, values, color=SERIES[0], width=0.58, zorder=3)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value * 1.06, f"{value:.0f}x",
                 ha="center", va="bottom", color=INK, fontsize=10)
    ax2.set_yscale("log")
    ax2.set_xticks(depths)
    ax2.set_ylim(1, max(values) * 3)
    figure.text(0.012, -0.06,
                "Serial chain, i.i.d. demand N(100, 20), exponential smoothing "
                "alpha = 0.3,\nbase-stock z = 1.645, 3-period lead time. "
                "30 replications, MSER-5 warm-up truncation.",
                color=INK_MUTED, fontsize=8)
    save(figure, "bullwhip_by_echelon.png")


def figure_disruption(warmup: int) -> None:
    """Retailer fill-rate recovery paths, disrupted against paired baseline."""
    start = 300
    config = merge_config(BASE, {"topology": {"capacity": 160.0}})
    scenarios = [
        ("supplier outage, 4 periods",
         {"disruptions": {"outages": [
             {"node": "source", "start": start, "duration": 4}]}}, start, 4),
        ("supplier outage, 8 periods",
         {"disruptions": {"outages": [
             {"node": "source", "start": start, "duration": 8}]}}, start, 8),
        ("demand shock, 2x for 4 periods",
         {"demand": {"shock": {
             "start": start, "duration": 4, "multiplier": 2.0}}}, start, 4),
    ]
    study = run_disruption_study(config, scenarios, warmup=warmup, horizon=60)

    figure = new_figure(9.5, 5.0)
    ax = figure.add_subplot(1, 1, 1)
    style_axis(ax, "periods since the disruption began",
               "retailer fill rate (%)",
               "A disruption outlasts itself by a factor of three")
    ax.axvspan(0, 8, color=GRID, zorder=1)
    ax.text(0.3, 30, "disruption\nwindow", color=INK_MUTED, fontsize=8,
            va="center")
    baseline = study.profiles[0].baseline_fill
    ax.plot(study.profiles[0].offsets, 100 * baseline, color=NEUTRAL,
            linewidth=1.4, linestyle=(0, (5, 3)), zorder=9,
            label="undisrupted baseline (same demand path)")
    for index, profile in enumerate(study.profiles):
        ax.plot(profile.offsets, 100 * profile.disrupted_fill,
                color=SERIES[index], linewidth=2.0, zorder=3 + index,
                label=profile.name)
        recovery = profile.recovery_offset.mean
        ax.plot([recovery], [100 * profile.disrupted_fill[int(recovery)]],
                marker="o", markersize=8, color=SERIES[index],
                markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=8)
        ax.annotate(f"back to baseline at +{recovery:.0f}",
                    xy=(recovery, 100 * profile.disrupted_fill[int(recovery)]),
                    xytext=(8, -16 - 15 * index), textcoords="offset points",
                    color=SERIES[index], fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=SERIES[index],
                                    linewidth=0.8, alpha=0.6))
    ax.set_ylim(0, 104)
    ax.set_xlim(0, 45)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    figure.text(0.012, -0.06,
                "Mean over 30 replications; each disrupted run is paired with its "
                "own undisrupted twin\nunder common random numbers. Factory "
                "capacity 160 units/period. Recovery is marked where\nsmoothed "
                "fill rate returns to within 1 point of the baseline and holds "
                "for 5 periods.",
                color=INK_MUTED, fontsize=8)
    save(figure, "disruption_recovery.png")


def figure_leadtime() -> None:
    """Inventory needed to hold 95% fill rate, by mean lead time and its CV."""
    cells = lead_time_grid(
        means=(2.0, 4.0, 8.0), cvs=(0.0, 0.25, 0.5), target_fill=0.95,
        base_config={"run": {"periods": 1040, "replications": 16, "seed": SEED,
                             "warmup": 80}},
    )
    lookup = {(c.mean_lead, c.cv_lead): c.inventory.mean for c in cells}
    means = [2.0, 4.0, 8.0]
    cvs = [0.0, 0.25, 0.5]

    figure = new_figure(9.0, 4.6)
    ax = figure.add_subplot(1, 1, 1)
    style_axis(ax, "mean lead time (periods)",
               "average inventory to hold 95% fill rate (units)",
               "Unreliability costs more than length")
    width = 0.26
    positions = np.arange(len(means))
    for index, cv in enumerate(cvs):
        offsets = positions + (index - 1) * (width + 0.03)
        values = [lookup[(mean, cv)] for mean in means]
        bars = ax.bar(offsets, values, width=width, color=ORDINAL[index],
                      zorder=3, label=f"lead-time CV {cv:g}")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.0f}",
                    ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{mean:g}" for mean in means])
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    reliable_long = lookup[(4.0, 0.0)]
    erratic_short = lookup[(2.0, 0.5)]
    ax.annotate(
        f"a reliable 4-period supplier ({reliable_long:.0f} units) beats an\n"
        f"erratic 2-period one ({erratic_short:.0f} units) at the same service",
        xy=(0.29, erratic_short + 8), xytext=(0.52, 243),
        color=INK_SOFT, fontsize=9, va="top",
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1.0),
    )
    ax.set_ylim(0, 300)
    figure.text(0.012, -0.06,
                "One stocking echelon fed by an always-available source. Every "
                "bar is calibrated by bisection\non the safety factor until the "
                "simulated fill rate is 95%, so the bars differ in inventory\n"
                "and not in service. 16 replications of 1040 periods.",
                color=INK_MUTED, fontsize=8)
    save(figure, "leadtime_tradeoff.png")


def figure_warmup() -> None:
    """System inventory transients and where MSER-5 truncates each of them."""
    variants = [
        ("4 periods opening cover, L = 3",
         {"initial_periods_of_stock": 4.0, "leadtime": {"mean": 2.0}}),
        ("empty pipeline, L = 3",
         {"initial_periods_of_stock": 0.0, "leadtime": {"mean": 2.0}}),
        ("empty pipeline, L = 7",
         {"initial_periods_of_stock": 0.0, "leadtime": {"mean": 6.0}}),
    ]
    figure = new_figure(9.5, 5.4)
    grid = figure.add_gridspec(3, 1, hspace=0.3)
    # Small multiples: the first two variants settle to the *same* operating
    # level, so on one axis the later curve simply paints over the earlier one
    # and the reader cannot tell whether a series is hidden or missing.
    axes = []
    for index, (label, override) in enumerate(variants):
        config = merge_config(BASE, override)
        paths = np.asarray([
            system_inventory_path(_replication(config, replication))
            for replication in range(20)
        ])
        full = paths.mean(axis=0)
        truncation = mser5_truncation(full)
        averaged = full[:300]
        ax = figure.add_subplot(grid[index, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        style_axis(ax, "", "", "")
        ax.axvspan(0, max(truncation, 1), color=GRID, zorder=1)
        ax.plot(np.arange(averaged.size), averaged, color=SERIES[index],
                linewidth=1.8, zorder=3)
        ax.axhline(full[truncation:].mean(), color=NEUTRAL, linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(80, 40000)
        ax.set_xlim(0, 300)
        ax.text(0.985, 0.86, label, transform=ax.transAxes, ha="right", va="top",
                color=SERIES[index], fontsize=10, fontweight="bold")
        ax.text(max(truncation, 1) + 6, 18000,
                f"MSER-5 truncates at {truncation}", color=INK_SOFT, fontsize=9,
                va="top")
        if index == 0:
            ax.set_title("MSER-5 finds the transient without anyone reading a chart",
                         color=INK, fontsize=12, loc="left", pad=10)
        if index < len(variants) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("period", color=INK_SOFT, fontsize=10)
    axes[1].set_ylabel("system inventory + backlog (units)", color=INK_SOFT,
                       fontsize=10)
    figure.text(0.012, -0.06,
                "Mean of 20 replications, log scale. Shaded band is the "
                "discarded warm-up; dashed line is\nthe mean after truncation. "
                "Averaging replications before applying MSER-5 is what makes\n"
                "the estimate stable -- on a single path the statistic returns "
                "zero on an obvious transient.",
                color=INK_MUTED, fontsize=8)
    save(figure, "warmup_truncation.png")


def main() -> int:
    os.makedirs(DOCS, exist_ok=True)
    warmup = estimate_warmup(BASE, pilots=4)
    print(f"warm-up truncation {warmup} periods")
    figure_bullwhip(warmup)
    figure_disruption(warmup)
    figure_leadtime()
    figure_warmup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
