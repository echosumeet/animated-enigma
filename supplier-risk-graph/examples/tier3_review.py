"""A tier-3 exposure review, the way it would be run before a sourcing council.

Order of the questions matters. Concentration first, because that is what the
category team already reports and believes. Then the hidden dependency, which is the
number that contradicts it. Then simulation and mitigation, so the meeting ends with
a ranked spend decision rather than a list of worries.
"""

from riskgraph import (
    expand,
    generate_network,
    geographic_hhi,
    hhi_by_tier,
    hidden_dependencies,
    rank_spofs,
    score_actions,
    simulate,
    supplier_hhi,
)


def money(x: float) -> str:
    return f"${x / 1e6:,.1f}M"


def main() -> None:
    net = generate_network(seed=7)
    fg = net.finished_goods()[0].part_id
    revenue = sum(p.annual_revenue for p in net.finished_goods())
    rows = expand(net, fg)

    print("1. Network")
    print(f"   {len(net.parts)} parts, {len(net.suppliers)} suppliers, {len(net.sites)} sites, "
          f"{len(net.bom)} BOM edges, {len(net.supply)} sourcing edges")
    print(f"   {fg} explodes to {len(rows)} parts over {max(r.depth for r in rows)} tiers; "
          f"portfolio revenue {money(revenue)}")

    print("\n2. Concentration as the category team reports it")
    for tier, h in hhi_by_tier(net, fg).items():
        print(f"   tier {tier}: HHI {h:.3f}  ({1 / h:.1f} effective suppliers)")
    geo, by_country = geographic_hhi(net, fg)
    top = ", ".join(f"{c} {s:.0%}" for c, s in list(by_country.items())[:3])
    print(f"   geographic HHI {geo:.3f} (supplier HHI {supplier_hhi(net, fg):.3f}); top countries: {top}")

    print("\n3. What tier-1 diversity hides")
    for h in hidden_dependencies(net, fg)[:3]:
        print(f"   {h.supplier_id} at depth {h.depth} in {h.countries}: reached through "
              f"{h.tier1_suppliers} of the tier-1 branches, {h.parts} parts, "
              f"{h.revenue_share:.0%} of revenue stops without it ({money(h.revenue_at_risk)})")

    print("\n4. Single points of failure, ranked by revenue at risk")
    for s in rank_spofs(net, top_n=5):
        flag = " articulation point" if s.articulation else ""
        print(f"   {s.node_id:24s} {money(s.revenue_at_risk):>10s} ({s.revenue_share:5.1%}) "
              f"recovery {s.mean_recovery_days:5.1f}d{flag}")

    print("\n5. Disruption simulation (2,000 trials, one-year horizon)")
    base = simulate(net, trials=2000, seed=11)
    print(f"   probability of any shortfall  {base.p_any_impact:.1%}")
    print(f"   expected annual loss          {money(base.expected_loss)}")
    print(f"   95th percentile loss          {money(base.p95_loss)}")
    print(f"   time to recover mean / p95    {base.mean_ttr_days:.0f}d / {base.p95_ttr_days:.0f}d")

    print("\n6. Mitigation, ranked by risk reduction per dollar")
    targets = [r.part_id for r in rows if r.sole_source and r.depth >= 2][:2]
    _, actions = score_actions(net, targets, trials=800, seed=11)
    print(f"   {'action':14s} {'target':10s} {'annual cost':>12s} {'reduction':>12s} {'per $1':>8s}")
    for a in actions:
        print(f"   {a.action:14s} {a.target:10s} {a.annual_cost:12,.0f} "
              f"{a.risk_reduction:12,.0f} {a.reduction_per_dollar:8.2f}")


if __name__ == "__main__":
    main()
