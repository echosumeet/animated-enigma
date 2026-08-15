"""Draw docs/pick_path.png: the layout, the pick density, and one picker tour.

    PYTHONPATH=src python benchmarks/make_figure.py

Two panels, same batch of orders, same router. Left is the warehouse as
received (SKUs in random feasible slots), right is the same warehouse after
velocity slotting. The rack faces are shaded by how often the held-out order
stream picks from them, so the picture shows the mechanism and not just the
outcome: slotting pulls the dark faces down to the depot end, and the tour
collapses onto them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from slotting import (  # noqa: E402
    CartCapacity,
    build_instance,
    random_assignment,
    savings_batching,
    two_opt,
    velocity_slotting,
)

INK = "#22303f"
MUTED = "#8a97a3"
PATH_C = "#c2451f"


def pick_density(instance, assignment) -> np.ndarray:
    """Picks per (aisle, bay) access point over the held-out stream."""
    c = instance.warehouse.config
    grid = np.zeros((c.n_aisles, c.n_bays))
    counts = instance.score_stream.line_counts()
    for sku, n in enumerate(counts):
        if n <= 0:
            continue
        loc = instance.warehouse.locations[int(assignment.location_of[sku])]
        grid[loc.aisle, loc.bay] += n
    return grid


def draw_panel(ax, instance, assignment, batch, title: str) -> float:
    w = instance.warehouse
    c = w.config
    density = pick_density(instance, assignment)
    vmax = max(density.max(), 1.0)
    cmap = plt.get_cmap("YlGnBu")
    half = c.aisle_pitch_m * 0.34

    for a in range(c.n_aisles):
        for b in range(c.n_bays):
            shade = cmap(0.12 + 0.88 * density[a, b] / vmax)
            for sign in (-1, 1):
                ax.add_patch(
                    Rectangle(
                        (w.aisle_x(a) + sign * half - (half * 0.62 if sign > 0 else 0.0),
                         b * c.bay_depth_m),
                        half * 0.62,
                        c.bay_depth_m * 0.92,
                        facecolor=shade,
                        edgecolor="white",
                        linewidth=0.25,
                    )
                )

    picks = batch.location_set()
    route = two_opt(w, picks)
    xs = [w.aisle_x(p.aisle) for p in route.waypoints]
    ys = [p.y for p in route.waypoints]
    ax.plot(xs, ys, color=PATH_C, linewidth=1.9, solid_capstyle="round", zorder=4)
    ax.scatter(
        [w.aisle_x(w.locations[p].aisle) for p in picks],
        [w.bay_y(w.locations[p].bay) for p in picks],
        s=26, facecolor="white", edgecolor=PATH_C, linewidth=1.4, zorder=5,
    )
    ax.scatter([w.aisle_x(w.depot.aisle)], [w.depot.y], marker="s", s=60,
               color=INK, zorder=6)
    ax.annotate("depot", (w.aisle_x(w.depot.aisle), w.depot.y), xytext=(6, -12),
                textcoords="offset points", fontsize=8, color=INK)

    ax.set_title(f"{title}\ntour = {route.distance:,.0f} m", fontsize=10, color=INK, pad=8)
    ax.set_xlim(-c.aisle_pitch_m * 0.8, (c.n_aisles - 1) * c.aisle_pitch_m + c.aisle_pitch_m * 0.8)
    ax.set_ylim(-2.5, w.aisle_length_m + 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("cross-aisle direction (m)", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    return route.distance


def main() -> None:
    instance = build_instance(seed=7)
    baseline = random_assignment(instance.constraints, seed=0)
    slotted = velocity_slotting(
        instance.constraints, metric="picks", pick_rate=instance.fit_rate
    )

    # One representative multi-order cart, picked under each layout.
    batches = savings_batching(
        instance.score_stream, slotted, instance.catalog, instance.warehouse, CartCapacity()
    )
    batch = max(batches, key=lambda b: b.lines)
    ids = set(batch.order_ids)
    orders = [o for o in instance.score_stream if o.order_id in ids]

    from slotting.batching import Batch

    def rebuild(assignment):
        picks = [int(assignment.location_of[s]) for o in orders for s in o.lines]
        return Batch(list(ids), picks, batch.lines, batch.cube)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.8))
    d0 = draw_panel(axes[0], instance, baseline, rebuild(baseline), "As received (random feasible)")
    d1 = draw_panel(axes[1], instance, slotted, rebuild(slotted), "Velocity slotted (ABC by picks)")

    axes[0].set_ylabel("along-aisle direction (m)", fontsize=8, color=MUTED)
    fig.suptitle(
        "One picker tour, same {} orders and {} lines, before and after slotting"
        "  —  {:.0f}% shorter".format(len(orders), batch.lines, 100 * (1 - d1 / d0)),
        fontsize=12, color=INK, y=0.99,
    )
    fig.text(
        0.5, 0.035,
        "Rack faces shaded by pick frequency over the held-out order stream (darker = faster moving). "
        "Route is 2-opt over the aisle-graph metric.",
        ha="center", fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    out = ROOT / "docs" / "pick_path.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"  as received {d0:,.0f} m -> slotted {d1:,.0f} m  ({100 * (1 - d1 / d0):.1f}% shorter)")


if __name__ == "__main__":
    main()
