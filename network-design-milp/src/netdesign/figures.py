"""Matplotlib figures. Imported only by the figure scripts, never by the tests.

Design rules applied here, so the plots read as one system:

* three categorical hues, assigned to entity type and never cycled (blue =
  plants, orange = DCs, aqua = suppliers); demand zones and flow lines are
  neutral context, not a fourth series;
* every coloured mark also carries a text label or a legend entry, so identity
  never depends on colour alone;
* one value axis per panel, recessive grid, no chart junk.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .instances import Instance
from .network_flow import NetworkSolution
from .reporting import cost_breakdown

__all__ = ["plot_network", "plot_tradeoffs", "SERIES"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e3e2dd"
SERIES = {"plant": "#2a78d6", "dc": "#eb6834", "supplier": "#1baf7a"}


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)


def _aspect(ax: Any, lats: Sequence[float]) -> None:
    """Approximate an equal-area look for a lat/lon scatter."""
    mid = sum(lats) / len(lats)
    ax.set_aspect(1.0 / max(0.2, math.cos(math.radians(mid))))


def plot_network(
    instance: Instance,
    solution: NetworkSolution,
    path: str,
    *,
    title: str | None = None,
) -> str:
    """Two-panel figure: the chosen network on a map, and where the money goes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.85, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    _style(ax2)

    # --- panel A: the network ------------------------------------------------
    open_dcs = set(solution.open_dcs)
    open_plants = set(solution.open_plants)

    max_units = max((r.units for r in solution.flows), default=1.0)
    for r in solution.flows:
        o, d = instance.site(r.origin), instance.site(r.dest)
        if d.kind == "zone":
            ax.plot(
                [o.lon, d.lon],
                [o.lat, d.lat],
                color=INK_MUTED,
                linewidth=0.4 + 2.2 * (r.units / max_units) ** 0.55,
                alpha=0.45,
                zorder=1,
                solid_capstyle="round",
            )
        elif d.kind == "dc":
            ax.plot(
                [o.lon, d.lon],
                [o.lat, d.lat],
                color=SERIES["plant"],
                linewidth=0.6 + 2.6 * (r.units / max_units) ** 0.55,
                alpha=0.5,
                zorder=2,
                solid_capstyle="round",
            )

    zone_units = [instance.zone_demand(z.id) for z in instance.zones]
    zmax = max(zone_units) or 1.0
    ax.scatter(
        [z.lon for z in instance.zones],
        [z.lat for z in instance.zones],
        s=[16 + 190 * (u / zmax) for u in zone_units],
        facecolor="#d8d7d1",
        edgecolor="#b7b6b0",
        linewidth=0.6,
        zorder=3,
        label="demand zone (area = volume)",
    )

    closed = [d for d in instance.dcs if d.id not in open_dcs]
    if closed:
        ax.scatter(
            [d.lon for d in closed],
            [d.lat for d in closed],
            marker="s",
            s=52,
            facecolor="none",
            edgecolor=INK_MUTED,
            linewidth=1.2,
            zorder=4,
        )
    used_dcs = [d for d in instance.dcs if d.id in open_dcs]
    ax.scatter(
        [d.lon for d in used_dcs],
        [d.lat for d in used_dcs],
        marker="s",
        s=132,
        color=SERIES["dc"],
        edgecolor=SURFACE,
        linewidth=1.6,
        zorder=6,
    )
    for d in used_dcs:
        ax.annotate(
            d.id,
            (d.lon, d.lat),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=8.5,
            color=INK,
            fontweight="bold",
            zorder=7,
        )

    plants = [p for p in instance.plants if p.id in open_plants]
    ax.scatter(
        [p.lon for p in plants],
        [p.lat for p in plants],
        marker="^",
        s=140,
        color=SERIES["plant"],
        edgecolor=SURFACE,
        linewidth=1.4,
        zorder=6,
    )
    for p in plants:
        ax.annotate(
            p.id,
            (p.lon, p.lat),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8.5,
            color=INK,
            zorder=7,
        )
    ax.scatter(
        [s.lon for s in instance.suppliers],
        [s.lat for s in instance.suppliers],
        marker="D",
        s=74,
        color=SERIES["supplier"],
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=6,
    )

    _aspect(ax, [z.lat for z in instance.zones])
    ax.set_xlabel("longitude", color=INK_2, fontsize=9)
    ax.set_ylabel("latitude", color=INK_2, fontsize=9)
    head = title or f"{instance.name}: cost-optimal network"
    ax.set_title(head, color=INK, fontsize=12, loc="left", pad=26)
    sub = (
        f"{len(solution.open_dcs)} of {len(instance.dcs)} DCs open  ·  "
        f"cost {solution.objective:,.0f}/period  ·  "
        f"demand-weighted last leg {solution.demand_weighted_km:,.0f} km"
    )
    ax.annotate(
        sub,
        xy=(0, 1.006),
        xycoords="axes fraction",
        fontsize=9,
        color=INK_2,
        ha="left",
        va="bottom",
    )

    handles = [
        Line2D([], [], marker="^", color="none", markerfacecolor=SERIES["plant"], markersize=9, label="plant (open)"),
        Line2D([], [], marker="s", color="none", markerfacecolor=SERIES["dc"], markersize=9, label="DC (open)"),
        Line2D([], [], marker="s", color="none", markerfacecolor="none", markeredgecolor=INK_MUTED, markersize=8, label="DC candidate (closed)"),
        Line2D([], [], marker="D", color="none", markerfacecolor=SERIES["supplier"], markersize=8, label="supplier"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#d8d7d1", markeredgecolor="#b7b6b0", markersize=9, label="demand zone (area = volume)"),
        Line2D([], [], color=INK_MUTED, linewidth=1.6, label="DC -> zone flow (width = volume)"),
        Line2D([], [], color=SERIES["plant"], linewidth=1.6, alpha=0.6, label="plant -> DC flow"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        ncol=2,
        handletextpad=0.6,
        columnspacing=1.4,
    )
    leg.set_zorder(8)

    # --- panel B: cost breakdown --------------------------------------------
    rows = cost_breakdown(solution)
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    shares = [r[2] for r in rows][::-1]
    bars = ax2.barh(
        range(len(values)),
        values,
        height=0.62,
        color=SERIES["plant"],
        edgecolor=SURFACE,
        linewidth=1.2,
    )
    for i, (b, v, sh) in enumerate(zip(bars, values, shares)):
        ax2.annotate(
            f"{v:,.0f}  ({sh:.0f}%)",
            (b.get_width(), i),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax2.set_xlim(0, max(values) * 1.42)
    ax2.set_xticks([])
    ax2.spines["bottom"].set_visible(False)
    ax2.set_title("cost per period, by bucket", color=INK, fontsize=12, loc="left", pad=26)
    ax2.annotate(
        f"total {sum(values):,.0f}",
        xy=(0, 1.006),
        xycoords="axes fraction",
        fontsize=9,
        color=INK_2,
        ha="left",
        va="bottom",
    )

    fig.tight_layout(pad=1.6)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_tradeoffs(
    curve: Sequence[tuple[int, float, float]],
    frequency: dict[str, float],
    stochastic: dict[str, float],
    path: str,
) -> str:
    """Three panels: the cost/#DC curve, network stability, and WS/RP/EEV.

    ``curve`` is ``(n_dcs, total_cost, demand_weighted_km)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)

    # panel 1: cost vs number of DCs
    ax = axes[0]
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    ax.plot(xs, ys, color=SERIES["plant"], linewidth=2.0, marker="o", markersize=6, zorder=3)
    best = min(range(len(ys)), key=lambda i: ys[i])
    ax.scatter([xs[best]], [ys[best]], s=150, color=SERIES["dc"], zorder=4, edgecolor=SURFACE, linewidth=1.6)
    span_y = max(ys) - min(ys)
    ax.set_ylim(min(ys) - 0.22 * span_y, max(ys) + 0.10 * span_y)
    ax.annotate(
        f"optimum: {xs[best]} DCs\n{ys[best]:,.0f}",
        (xs[best], ys[best]),
        textcoords="offset points",
        xytext=(12, -10),
        va="top",
        fontsize=9,
        color=INK,
    )
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("number of DCs open", color=INK_2, fontsize=9)
    ax.set_ylabel("total cost per period", color=INK_2, fontsize=9)
    ax.set_title("the curve is flat near the optimum", color=INK, fontsize=11, loc="left", pad=22)
    ax.set_xticks(xs)

    # panel 2: how often each DC opens across the sensitivity sweep
    ax = axes[1]
    items = sorted(frequency.items(), key=lambda kv: kv[1])
    ax.barh(
        range(len(items)),
        [100 * v for _, v in items],
        height=0.62,
        color=[SERIES["dc"] if v >= 0.9 else ("#f0a684" if v > 0.05 else "#d8d7d1") for _, v in items],
        edgecolor=SURFACE,
        linewidth=1.2,
    )
    for i, (_, v) in enumerate(items):
        ax.annotate(
            f"{100 * v:.0f}%",
            (100 * v, i),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([k for k, _ in items], fontsize=9, color=INK)
    ax.set_xlim(0, 118)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("share of sweep runs each DC opens", color=INK, fontsize=11, loc="left", pad=22)
    ax.annotate(
        "solid = core (>=90%)   ·   light = swing   ·   grey = never",
        xy=(0, 1.006),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_2,
        va="bottom",
    )

    # panel 3: WS / RP / EEV with VSS and EVPI called out
    ax = axes[2]
    keys = ["ws", "rp", "eev"]
    labels = ["WS\n(perfect info)", "RP\n(stochastic)", "EEV\n(mean-value design)"]
    vals = [stochastic[k] for k in keys]
    colors = ["#d8d7d1", SERIES["plant"], SERIES["dc"]]
    ax.bar(range(3), vals, width=0.56, color=colors, edgecolor=SURFACE, linewidth=1.4, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(
            f"{v:,.0f}",
            (i, v),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
            color=INK,
        )
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    ax.set_ylim(lo - 0.55 * span, hi + 0.30 * span)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("expected cost per period", color=INK_2, fontsize=9)
    ax.set_title("what uncertainty is worth", color=INK, fontsize=11, loc="left", pad=22)
    ax.annotate(
        f"VSS = EEV - RP = {stochastic['eev'] - stochastic['rp']:,.0f}\n"
        f"EVPI = RP - WS = {stochastic['rp'] - stochastic['ws']:,.0f}",
        xy=(0.02, 0.50),
        xycoords="axes fraction",
        fontsize=9,
        color=INK,
        va="bottom",
    )

    fig.tight_layout(pad=1.8)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path
