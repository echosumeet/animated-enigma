"""End-to-end walkthrough: build a warehouse, slot it, route it, batch it.

    PYTHONPATH=src python examples/slot_and_route.py

Runs in a few seconds on a laptop. Everything is generated from a seed, so
there is no data file to fetch and the output is reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from slotting import (  # noqa: E402
    CartCapacity,
    CatalogConfig,
    OrderConfig,
    ROUTERS,
    WarehouseConfig,
    build_instance,
    evaluate_batches,
    evaluate_travel,
    exact_aisle_dp,
    random_assignment,
    savings_batching,
    single_order_batches,
    velocity_slotting,
)

RULE = "-" * 72


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    # ------------------------------------------------------------------
    section("1. Build an instance")
    # A 8 x 20 x 4 forward pick area, 600 SKUs, 3,000 orders. Demand is Zipf,
    # baskets are overdispersed and family-themed.
    instance = build_instance(
        seed=42,
        warehouse_config=WarehouseConfig(n_aisles=8, n_bays=20, n_levels=4),
        catalog_config=CatalogConfig(n_skus=600, n_families=14),
        order_config=OrderConfig(n_orders=3_000),
    )
    d = instance.describe()
    print(f"{int(d['locations']):,} pick faces over a {d['aisle_length_m']:.1f} m aisle run")
    print(f"{int(d['skus']):,} SKUs, {int(d['orders']):,} orders, "
          f"{d['mean_lines']:.2f} lines per order")
    print(f"fastest 20% of SKUs carry {100 * d['top20pct_pick_share']:.1f}% of picks")
    print(f"{int(d['hazmat_skus'])} hazmat SKUs are restricted to floor level")
    print(f"slotting sees {len(instance.fit_stream):,} orders; "
          f"travel is scored on the {len(instance.score_stream):,} it has never seen")

    # ------------------------------------------------------------------
    section("2. Slot it")
    baseline = random_assignment(instance.constraints, seed=0)
    slotted = velocity_slotting(
        instance.constraints, metric="picks", pick_rate=instance.fit_rate
    )
    for name, a in (("as received", baseline), ("velocity slotted", slotted)):
        problems = instance.constraints.violations(a)
        print(f"{name:18s} complete={a.is_complete()} violations={len(problems)}")

    w = instance.warehouse
    depot_d = w.depot_distance_vector()
    for name, a in (("as received", baseline), ("velocity slotted", slotted)):
        weighted = float(
            np.average(depot_d[a.location_of], weights=np.maximum(instance.fit_rate, 1e-9))
        )
        print(f"{name:18s} pick-weighted mean distance from depot = {weighted:5.1f} m")

    # ------------------------------------------------------------------
    section("3. Route it")
    policies = ("s_shape", "return", "midpoint", "largest_gap", "two_opt")
    before = evaluate_travel(instance, baseline, policies=policies)
    after = evaluate_travel(instance, slotted, policies=policies)
    n = len(instance.score_stream)
    print(f"{'router':16s}{'as received':>16s}{'slotted':>14s}{'cut':>9s}")
    for p in policies:
        print(f"{p:16s}{before[p] / n:13.1f} m{after[p] / n:11.1f} m"
              f"{100 * (1 - after[p] / before[p]):8.1f}%")
    print("\nMetres per order, held-out stream. Note that the ranking of routers")
    print("changes once the warehouse is slotted: aisle-traversing policies lose")
    print("their edge when the picks all sit in the first few bays.")

    # ------------------------------------------------------------------
    section("4. How far off optimal is the routing?")
    rng = np.random.default_rng(0)
    orders = list(instance.score_stream)
    sample = rng.choice(len(orders), size=150, replace=False)
    gaps: dict[str, list[float]] = {p: [] for p in policies}
    for i in sample:
        picks = sorted({int(slotted.location_of[s]) for s in orders[int(i)].lines})
        opt = exact_aisle_dp(instance.warehouse, picks)
        if opt <= 0:
            continue
        for p in policies:
            gaps[p].append(ROUTERS[p](instance.warehouse, picks).distance / opt - 1.0)
    print(f"{'router':16s}{'mean gap':>11s}{'p90 gap':>11s}   (vs exact aisle DP, 150 tours)")
    for p in policies:
        print(f"{p:16s}{100 * float(np.mean(gaps[p])):10.2f}%"
              f"{100 * float(np.percentile(gaps[p], 90)):10.2f}%")

    # ------------------------------------------------------------------
    section("5. Batch it")
    cap = CartCapacity()
    single = evaluate_batches(
        instance.warehouse,
        single_order_batches(instance.score_stream, slotted, instance.catalog),
        "two_opt", "single order",
    )
    batched = evaluate_batches(
        instance.warehouse,
        savings_batching(
            instance.score_stream, slotted, instance.catalog, instance.warehouse, cap
        ),
        "two_opt", "savings batching",
    )
    print(f"cart limits: {cap.max_orders} orders / {cap.max_lines} lines / {cap.max_cube_m3} m3")
    print(f"{'policy':20s}{'tours':>8s}{'total km':>11s}{'m/order':>10s}{'aisles/tour':>13s}")
    for r in (single, batched):
        print(f"{r.policy:20s}{r.n_batches:8,d}{r.total_distance_m / 1000:11.1f}"
              f"{r.distance_per_order_m:10.1f}{r.mean_aisles_per_batch:13.1f}")
    print(f"\nbatching cuts a further {100 * (1 - batched.total_distance_m / single.total_distance_m):.1f}% "
          "of travel on top of the slotting")

    total_cut = 1 - batched.total_distance_m / (before["two_opt"])
    print(f"\nslotting + batching together: {100 * total_cut:.1f}% below the as-received, "
          "one-order-per-tour baseline")


if __name__ == "__main__":
    main()
