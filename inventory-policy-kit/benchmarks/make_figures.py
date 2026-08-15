"""Generate the figures in docs/ by actually running the library.

    PYTHONPATH=src python benchmarks/make_figures.py

Four figures, each answering one question a reader will have:

    docs/service_definitions.png  - how far apart are CSL and fill-rate sizing?
    docs/efficient_frontier.png   - what does the next point of service cost?
    docs/validation.png           - do the formulas survive simulation?
    docs/placement.png            - where does the guaranteed-service model put stock?
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

from invkit.distributions import GammaLTD, MixtureLTD  # noqa: E402
from invkit.frontier import exchange_curve, marginal_cost_of_service  # noqa: E402
from invkit.guaranteed_service import example_bom_tree, solve_guaranteed_service  # noqa: E402
from invkit.leadtime import LeadTimeSpec, ltd_with_undershoot, undershoot_moments  # noqa: E402
from invkit.policies import SQPolicy, build_RS  # noqa: E402
from invkit.safety_stock import (  # noqa: E402
    compare_service_definitions,
    ss_from_cycle_service_level,
    ss_from_fill_rate,
)
from invkit.simulation import DemandProcess, simulate_policy  # noqa: E402

DEMAND_MEAN, DEMAND_SD, LEAD_TIME = 100.0, 30.0, 5
UNIT_COST, HOLDING_RATE = 25.0, 0.25 / 365.0

# A small, deliberately restrained palette: one accent per series, one neutral
# for reference lines and annotation. Anything more competes with the data.
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
BLUE = "#2c6fbb"
AMBER = "#c9721a"
TEAL = "#2a8a7e"
ROSE = "#a6355b"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
    }
)


def _ltd():
    spec = LeadTimeSpec.deterministic(DEMAND_MEAN, DEMAND_SD, LEAD_TIME)
    return ltd_with_undershoot(spec, DEMAND_MEAN, DEMAND_SD)


def fig_service_definitions() -> Path:
    ltd = _ltd()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax = axes[0]
    targets = np.linspace(0.80, 0.995, 60)
    Q = 800.0
    csl_ss = [ss_from_cycle_service_level(ltd, t, Q).safety_stock for t in targets]
    fill_ss = [ss_from_fill_rate(ltd, t, Q).safety_stock for t in targets]
    ax.plot(targets, csl_ss, color=BLUE, lw=2.0, label="read as cycle service level")
    ax.plot(targets, fill_ss, color=AMBER, lw=2.0, label="read as fill rate")
    ax.fill_between(targets, fill_ss, csl_ss, color=BLUE, alpha=0.07, lw=0)
    ax.axhline(0.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("service target as written in the planning parameter")
    ax.set_ylabel("safety stock (units)")
    ax.set_title("The same target, two definitions")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    c = compare_service_definitions(ltd, 0.95, Q)
    ax.annotate(
        f"at 0.95: {c['ss_delta']:,.0f} units apart",
        xy=(0.95, (c["ss_csl_basis"] + c["ss_fill_basis"]) / 2),
        xytext=(0.815, 175),
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
    )

    ax = axes[1]
    q_over_sigma = np.linspace(1.0, 45.0, 80)
    for target, colour in ((0.95, BLUE), (0.98, TEAL), (0.99, ROSE)):
        deltas = []
        for r in q_over_sigma:
            Q = float(r * ltd.sd)
            c = compare_service_definitions(ltd, target, Q)
            deltas.append(c["ss_delta"])
        ax.plot(q_over_sigma, deltas, color=colour, lw=2.0, label=f"target {target:.2f}")
    ax.set_xlabel("order quantity / sd of lead-time demand")
    ax.set_ylabel("extra safety stock from reading the target as CSL (units)")
    ax.set_title("The gap is driven by lot size, which CSL ignores")
    ax.grid(axis="y")
    ax.legend(loc="upper left")

    fig.suptitle(
        "Cycle service level vs fill rate  |  gamma lead-time demand, mean 555, sd 76",
        fontsize=11.5,
        y=1.005,
        color=INK,
    )
    fig.tight_layout()
    path = DOCS / "service_definitions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_efficient_frontier() -> Path:
    ltd = _ltd()
    Q = 800.0
    # Dense grid for a smooth frontier; the marginal-cost panel reuses the same
    # 14-point grid the benchmark table reports, so the two agree exactly.
    fill_pts = exchange_curve(ltd, Q, UNIT_COST, HOLDING_RATE, n_points=60, lo=0.80,
                              hi=0.9985)["fill"]
    coarse = exchange_curve(ltd, Q, UNIT_COST, HOLDING_RATE, n_points=14, lo=0.80,
                            hi=0.995)["fill"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))

    # --- left: the frontier itself, with the two readings of "95%" marked ------
    ax = axes[0]
    xs = [100 * p.achieved_fill for p in fill_pts]
    ys = [p.holding_cost for p in fill_pts]
    ax.plot(xs, ys, color=TEAL, lw=2.4, label="efficient frontier (sized on fill rate)")
    ax.fill_between(xs, min(ys) - 0.4, ys, color=TEAL, alpha=0.06, lw=0)

    fill95 = ss_from_fill_rate(ltd, 0.95, Q)
    csl95 = ss_from_cycle_service_level(ltd, 0.95, Q)
    h = UNIT_COST * HOLDING_RATE
    marks = [
        ("target 0.95 read as fill rate", fill95, AMBER),
        ("target 0.95 read as cycle service", csl95, BLUE),
    ]
    for label, res, colour in marks:
        from invkit.safety_stock import fill_rate_of_sQ

        x = 100 * fill_rate_of_sQ(ltd, res.reorder_point, Q)
        y = h * (Q / 2.0 + res.safety_stock)
        ax.plot(x, y, "o", color=colour, ms=9, mec="white", mew=1.2, zorder=4, label=label)

    x_fill = 100 * 0.95
    y_fill = h * (Q / 2.0 + fill95.safety_stock)
    y_csl = h * (Q / 2.0 + csl95.safety_stock)
    ax.annotate(
        "",
        xy=(x_fill, y_csl), xytext=(x_fill, y_fill),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1),
    )
    from invkit.safety_stock import fill_rate_of_sQ as _fill

    delivered_csl = 100 * _fill(ltd, csl95.reorder_point, Q)
    ax.annotate(
        f"+{y_csl - y_fill:.2f}/day\nfor +{delivered_csl - 95:.1f} points\nnobody asked for",
        xy=(x_fill, 0.5 * (y_fill + y_csl)), xytext=(10, -8),
        textcoords="offset points", fontsize=8.5, color=INK,
    )
    ax.set_xlabel("fill rate actually delivered (%)")
    ax.set_ylabel("inventory holding cost per day")
    ax.set_title("Both readings are on the frontier. Only one was chosen.")
    ax.set_xlim(94.6, 100.05)
    ax.grid(axis="y")
    ax.legend(loc="lower right")

    # --- right: marginal cost of the next point of fill rate ------------------
    ax = axes[1]
    rows = marginal_cost_of_service(coarse)
    mid = [100 * 0.5 * (r["from_fill"] + r["to_fill"]) for r in rows]
    vals = [r["cost_per_service_point"] for r in rows]
    ax.plot(mid, vals, color=AMBER, lw=2.4)
    ax.fill_between(mid, 0, vals, color=AMBER, alpha=0.08, lw=0)
    ax.set_xlabel("fill rate (%)")
    ax.set_ylabel("holding cost per additional point of fill rate")
    ax.set_title("The curve steepens: what the next point costs")
    ax.set_xlim(94.6, 100.05)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y")
    ratio = vals[-1] / vals[0]
    ax.annotate(
        f"the last point costs {ratio:.0f}x the first",
        xy=(mid[-1], vals[-1]), xytext=(-165, -14), textcoords="offset points",
        fontsize=8.5, color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
    )

    fig.suptitle(
        "Cost-service frontier  |  unit cost 25, holding 25%/yr, lot 800 units",
        fontsize=11.5, y=1.005, color=INK,
    )
    fig.tight_layout()
    path = DOCS / "efficient_frontier.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_validation() -> Path:
    """Analytic target on x, simulated outcome on y. Points should sit on y = x."""
    proc = DemandProcess(DEMAND_MEAN, DEMAND_SD, "gamma")
    ltd = _ltd()
    Q = 800.0
    n_periods, warmup = 200_000, 2_000

    series: dict[str, tuple[list[float], list[float], str, str]] = {}

    xs, ys = [], []
    for t in (0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
        res = ss_from_cycle_service_level(ltd, t, Q)
        sim = simulate_policy(SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
                              n_periods=n_periods, warmup=warmup, seed=101)
        xs.append(t)
        ys.append(sim.cycle_service_level)
    series["(s,Q) cycle service"] = (xs, ys, BLUE, "o")

    xs, ys = [], []
    for t in (0.95, 0.97, 0.98, 0.99, 0.995):
        res = ss_from_fill_rate(ltd, t, Q)
        sim = simulate_policy(SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
                              n_periods=n_periods, warmup=warmup, seed=303)
        xs.append(t)
        ys.append(sim.fill_rate)
    series["(s,Q) fill rate"] = (xs, ys, AMBER, "s")

    xs, ys = [], []
    for t in (0.90, 0.95, 0.98, 0.99):
        policy, _ = build_RS(LeadTimeSpec.deterministic(DEMAND_MEAN, DEMAND_SD, 3), R=1,
                             target_csl=t)
        sim = simulate_policy(policy, proc, {3: 1.0}, n_periods=n_periods, warmup=warmup, seed=505)
        xs.append(t)
        ys.append(sim.ready_rate)
    series["(R,S) R=1 ready rate"] = (xs, ys, TEAL, "^")

    xs, ys = [], []
    for t in (0.95, 0.98, 0.99):
        policy, _ = build_RS(LeadTimeSpec.deterministic(DEMAND_MEAN, DEMAND_SD, 3), R=5,
                             target_fill=t)
        sim = simulate_policy(policy, proc, {3: 1.0}, n_periods=n_periods, warmup=warmup, seed=606)
        xs.append(t)
        ys.append(sim.fill_rate)
    series["(R,S) R=5 fill rate"] = (xs, ys, ROSE, "D")

    # The control: the same target sized without the undershoot correction.
    naive_ltd = GammaLTD.from_moments(DEMAND_MEAN * LEAD_TIME, DEMAND_SD * math.sqrt(LEAD_TIME))
    xs, ys = [], []
    for t in (0.95, 0.98):
        res = ss_from_cycle_service_level(naive_ltd, t, Q)
        sim = simulate_policy(SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
                              n_periods=n_periods, warmup=warmup, seed=101)
        xs.append(t)
        ys.append(sim.cycle_service_level)
    naive = (xs, ys)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.plot([0.86, 1.0], [0.86, 1.0], color=MUTED, lw=1.0, ls="--", label="perfect agreement")
    for label, (xs, ys, colour, marker) in series.items():
        ax.plot(xs, ys, marker, color=colour, ms=7, label=label, mec="white", mew=0.8)
    ax.plot(*naive, "x", color=INK, ms=8, mew=1.6,
            label="(s,Q) CSL, undershoot ignored")
    ax.set_xlabel("analytic target")
    ax.set_ylabel("simulated outcome (200,000 days)")
    ax.set_title("Every formula, checked against the policy it produces")
    ax.set_xlim(0.885, 1.0)
    ax.set_ylim(0.76, 1.005)
    ax.grid()
    ax.legend(loc="lower right")
    ax.annotate(
        "ignoring undershoot costs\n15.7 points of cycle service",
        xy=(0.95, naive[1][0]), xytext=(0.897, 0.835),
        fontsize=8.5, color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
    )
    fig.tight_layout()
    path = DOCS / "validation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_placement() -> Path:
    import networkx as nx

    tree = example_bom_tree()
    res = solve_guaranteed_service(tree, 0.95, 0.25)

    G = nx.DiGraph()
    for name in tree.stages:
        G.add_node(name)
    for (supplier, customer), usage in tree.arcs.items():
        G.add_edge(supplier, customer, usage=usage)

    # Layer by longest path from a source, so material flows left to right.
    depth: dict[str, int] = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        depth[node] = 0 if not preds else max(depth[p] for p in preds) + 1
    layers: dict[int, list[str]] = {}
    for n, d in depth.items():
        layers.setdefault(d, []).append(n)
    pos = {}
    for d, nodes in layers.items():
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (d, -(i - (len(nodes) - 1) / 2.0) * 1.85)

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=MUTED, width=1.2, arrowsize=13,
        node_size=2600, connectionstyle="arc3,rad=0.02",
    )
    costs = res.safety_stock_cost
    max_cost = max(costs.values()) or 1.0
    for n in G.nodes:
        holds = res.net_replenishment_times[n] > 0
        size = 900 + 2600 * (costs[n] / max_cost)
        ax.scatter(
            *pos[n], s=size,
            color=AMBER if holds else "white",
            edgecolors=AMBER if holds else MUTED,
            linewidths=1.6, zorder=3, alpha=0.9 if holds else 1.0,
        )
    for n in G.nodes:
        tau = res.net_replenishment_times[n]
        label = f"{n}\nT={tree.stages[n].processing_time}  S={res.service_times[n]}"
        if tau > 0:
            label += f"\ntau={tau}  SS={res.safety_stock[n]:,.0f}"
        else:
            label += "\nno safety stock"
        size = 900 + 2600 * (costs[n] / max_cost)
        radius_pts = math.sqrt(size / math.pi)
        ax.annotate(
            label, pos[n], textcoords="offset points", xytext=(0, -(radius_pts + 16)),
            ha="center", va="top", fontsize=7.6, color=INK,
        )
    ax.scatter([], [], s=160, color=AMBER, edgecolors=AMBER,
               label="decoupling point (holds safety stock; area = holding cost)")
    ax.scatter([], [], s=160, color="white", edgecolors=MUTED, label="pass-through (tau = 0)")
    ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(-0.02, 1.02))
    ax.set_title(
        "Guaranteed-service placement: 6 of 9 stages carry the buffer, 3 are pass-through\n"
        f"total safety-stock holding cost {res.total_cost:,.0f} per period at a 95% demand bound",
        fontsize=11,
    )
    ax.set_axis_off()
    ax.margins(x=0.08, y=0.30)
    fig.tight_layout()
    path = DOCS / "placement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    for fn in (fig_service_definitions, fig_efficient_frontier, fig_validation, fig_placement):
        path = fn()
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
