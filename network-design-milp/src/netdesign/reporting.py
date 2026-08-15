"""Turning a solved model into something a network team will argue with.

The optimiser returns one number. A network review needs six: what opened, what
it costs by bucket, how hard each site is working, how far the average unit
travels on the last leg, who is single-sourced, and where the answer is fragile.
Anything less and the meeting reverts to opinions about the objective value.

Two of these deserve comment.

*Utilisation* is reported against the throughput ceiling **and** against the
minimum-volume threshold, because a DC sitting at 36% of capacity but exactly
on its minimum volume is not a mistake - it is a facility the model kept open
for coverage and is holding at the contractual floor. Without both numbers that
reads as an error and someone "fixes" it.

*Demand-weighted last-leg distance* is the service proxy. Cost per unit hides
the difference between a network that is cheap because it is dense and one that
is cheap because it abandoned the far zones.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .instances import Instance
from .network_flow import FlowRecord, NetworkSolution

__all__ = [
    "cost_breakdown",
    "facility_table",
    "flow_by_echelon",
    "service_profile",
    "format_report",
    "markdown_table",
]

_COST_LABELS = {
    "plant_fixed": "plant fixed cost",
    "dc_fixed": "DC fixed cost",
    "production": "production (variable)",
    "handling": "DC handling",
    "transport_inbound": "transport: supplier -> plant",
    "transport_primary": "transport: plant -> DC",
    "transport_outbound": "transport: DC -> zone",
    "unmet_penalty": "unmet demand penalty",
}


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Minimal GitHub-flavoured markdown table with right-aligned numbers."""
    n = len(headers)
    body = [list(r)[:n] + [""] * max(0, n - len(list(r))) for r in rows]
    aligns = []
    for j in range(n):
        numeric = (
            all(isinstance(r[j], (int, float)) or _looks_numeric(r[j]) for r in body)
            if body
            else False
        )
        aligns.append("---:" if numeric else ":---")
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join(aligns) + "|"]
    for r in body:
        out.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(out)


def _looks_numeric(v: object) -> bool:
    s = str(v).replace(",", "").replace("%", "").replace("-", "").strip()
    if not s:
        return False
    try:
        float(s)
    except ValueError:
        return False
    return True


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def cost_breakdown(solution: NetworkSolution) -> list[tuple[str, float, float]]:
    """``(label, cost, share_of_total)`` sorted descending, zero buckets dropped."""
    total = sum(v for v in solution.costs.values())
    rows = [
        (_COST_LABELS.get(k, k), float(v), (100.0 * v / total) if total else 0.0)
        for k, v in solution.costs.items()
        if abs(v) > 1e-6
    ]
    rows.sort(key=lambda r: -r[1])
    return rows


def facility_table(solution: NetworkSolution, instance: Instance) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fac in (*instance.plants, *instance.dcs):
        u = solution.utilization.get(fac.id)
        if u is None:
            continue
        is_open = bool(u["open"])
        if not is_open and u["throughput"] <= 0:
            continue
        rows.append(
            {
                "id": fac.id,
                "kind": fac.kind,
                "name": fac.name,
                "open": is_open,
                "throughput": u["throughput"],
                "capacity": u["capacity"],
                "utilization_pct": 100.0 * u["utilization"],
                "min_volume": u["min_volume"],
                "at_min_volume": bool(
                    u["min_volume"] > 0 and u["throughput"] <= u["min_volume"] * 1.0001
                ),
                "fixed_cost": fac.fixed_cost,
            }
        )
    rows.sort(key=lambda r: (r["kind"], -float(r["throughput"])))
    return rows


def flow_by_echelon(
    solution: NetworkSolution, instance: Instance
) -> list[dict[str, object]]:
    """Units, weight-km and cost aggregated by echelon and mode."""
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"units": 0.0, "kg_km": 0.0, "cost": 0.0, "lanes": 0.0}
    )
    kg = {c.id: c.kg_per_unit for c in instance.commodities}
    for r in solution.flows:
        ech = f"{instance.site(r.origin).kind} -> {instance.site(r.dest).kind}"
        row = agg[(ech, r.mode)]
        row["units"] += r.units
        row["kg_km"] += r.units * kg[r.commodity] * r.distance_km
        row["cost"] += r.cost
        row["lanes"] += 1
    order = {"supplier -> plant": 0, "plant -> dc": 1, "dc -> zone": 2}
    out = [
        {
            "echelon": ech,
            "mode": mode,
            "lanes": int(v["lanes"]),
            "units": v["units"],
            "kg_km": v["kg_km"],
            "cost": v["cost"],
            "cost_per_unit": v["cost"] / v["units"] if v["units"] else 0.0,
        }
        for (ech, mode), v in agg.items()
    ]
    out.sort(key=lambda r: (order.get(str(r["echelon"]), 9), str(r["mode"])))
    return out


def service_profile(solution: NetworkSolution, instance: Instance) -> dict[str, float]:
    """Distance and sourcing-concentration statistics for the last leg."""
    last_leg = [r for r in solution.flows if instance.site(r.dest).kind == "zone"]
    units = sum(r.units for r in last_leg)
    if units <= 0:
        return {"demand_weighted_km": 0.0, "p90_km": 0.0, "single_sourced_zones": 0.0}
    ordered = sorted(last_leg, key=lambda r: r.distance_km)
    cum = 0.0
    p90 = ordered[-1].distance_km
    for r in ordered:
        cum += r.units
        if cum >= 0.9 * units:
            p90 = r.distance_km
            break
    single = sum(1 for mix in solution.zone_service.values() if len(mix) == 1)
    longest = max(last_leg, key=lambda r: r.distance_km)
    return {
        "demand_weighted_km": solution.demand_weighted_km,
        "p90_km": float(p90),
        "max_km": float(longest.distance_km),
        "single_sourced_zones": float(single),
        "zones_served": float(len(solution.zone_service)),
        "avg_dcs_per_zone": float(
            sum(len(m) for m in solution.zone_service.values()) / max(1, len(solution.zone_service))
        ),
    }


def format_report(solution: NetworkSolution, instance: Instance, *, title: str = "") -> str:
    """The full text report - what gets pasted into the decision document."""
    lines: list[str] = []
    head = title or f"network design report - {instance.name}"
    lines.append(head)
    lines.append("=" * len(head))
    lines.append(f"status              {solution.status}")
    if solution.objective is not None:
        lines.append(f"total cost/period   {solution.objective:,.0f}")
    lines.append(f"solve time          {solution.runtime:.2f}s")
    if solution.model_stats:
        s = solution.model_stats
        lines.append(
            f"model size          {s['variables']:,} vars ({s['binaries']:,} integer), "
            f"{s['constraints']:,} rows, {s['nonzeros']:,} nonzeros"
        )
    if not solution.is_optimal:
        return "\n".join(lines)

    lines.append("")
    lines.append(f"open plants ({len(solution.open_plants)}): {', '.join(solution.open_plants)}")
    lines.append(f"open DCs    ({len(solution.open_dcs)}): {', '.join(solution.open_dcs)}")

    lines.append("")
    lines.append("cost breakdown")
    lines.append(f"{'bucket':<32}{'cost/period':>16}{'share':>9}")
    for label, cost, share in cost_breakdown(solution):
        lines.append(f"{label:<32}{cost:>16,.0f}{share:>8.1f}%")
    total = sum(solution.costs.values())
    lines.append(f"{'TOTAL':<32}{total:>16,.0f}{100.0:>8.1f}%")

    lines.append("")
    lines.append("facilities")
    lines.append(
        f"{'id':<7}{'kind':<9}{'throughput':>13}{'capacity':>13}{'util':>8}{'note':>16}"
    )
    for row in facility_table(solution, instance):
        note = "at min volume" if row["at_min_volume"] else ("open" if row["open"] else "closed")
        lines.append(
            f"{row['id']:<7}{row['kind']:<9}{float(row['throughput']):>13,.0f}"
            f"{float(row['capacity']):>13,.0f}{float(row['utilization_pct']):>7.1f}%{note:>16}"
        )

    lines.append("")
    lines.append("flow by echelon")
    lines.append(f"{'echelon':<20}{'mode':<9}{'lanes':>7}{'units':>13}{'cost':>14}{'$/unit':>9}")
    for row in flow_by_echelon(solution, instance):
        lines.append(
            f"{str(row['echelon']):<20}{str(row['mode']):<9}{int(row['lanes']):>7}"
            f"{float(row['units']):>13,.0f}{float(row['cost']):>14,.0f}"
            f"{float(row['cost_per_unit']):>9.2f}"
        )

    prof = service_profile(solution, instance)
    lines.append("")
    lines.append("service profile (last leg, DC -> zone)")
    lines.append(f"  demand-weighted distance   {prof['demand_weighted_km']:,.0f} km")
    lines.append(f"  p90 distance               {prof['p90_km']:,.0f} km")
    lines.append(f"  longest lane used          {prof['max_km']:,.0f} km")
    lines.append(
        f"  single-sourced zones       {int(prof['single_sourced_zones'])} of "
        f"{int(prof['zones_served'])}"
    )
    lines.append(f"  average DCs per zone       {prof['avg_dcs_per_zone']:.2f}")
    if solution.unmet_units > 0:
        lines.append(f"  UNMET DEMAND               {solution.unmet_units:,.0f} units")
    return "\n".join(lines)


def top_flows(solution: NetworkSolution, n: int = 15) -> list[FlowRecord]:
    return solution.flows[:n]
