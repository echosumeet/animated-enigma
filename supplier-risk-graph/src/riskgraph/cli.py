"""Command line entry point: python -m riskgraph <command>."""

from __future__ import annotations

import argparse
import json

from .concentration import geographic_hhi, hhi_by_tier, hidden_dependencies, supplier_hhi
from .figures import spof_figure
from .flow import expand
from .generate import generate_network
from .model import load_network
from .mitigation import score_actions
from .simulate import simulate
from .spof import rank_spofs


def _network(args):
    return load_network(args.network) if args.network else generate_network(seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="riskgraph", description="Multi-tier supplier risk analysis")
    ap.add_argument("command", choices=["generate", "expand", "spof", "concentration", "simulate", "mitigate", "figure"])
    ap.add_argument("--network", help="path to a network JSON file (default: generate one)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="network.json")
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args(argv)

    if args.command == "generate":
        net = generate_network(seed=args.seed)
        net.to_json(args.out)
        print(f"wrote {args.out}: {len(net.parts)} parts, {len(net.sites)} sites, {len(net.suppliers)} suppliers")
        return 0

    net = _network(args)
    fg = net.finished_goods()[0].part_id

    if args.command == "expand":
        for row in expand(net, fg)[: args.top * 4]:
            print(f"{'  ' * row.depth}{row.part_id:10s} depth={row.depth} qty/fg={row.qty_per_fg:8.2f} "
                  f"spend={row.annual_spend:14,.0f} sole_source={row.sole_source}")
    elif args.command == "spof":
        for r in rank_spofs(net, top_n=args.top):
            print(f"{r.node_id:22s} rar={r.revenue_at_risk:14,.0f} ({r.revenue_share:6.1%}) "
                  f"articulation={r.articulation} sole_source_parts={r.sole_source_parts}")
    elif args.command == "concentration":
        gh, by_country = geographic_hhi(net, fg)
        print(json.dumps({
            "hhi_by_tier": hhi_by_tier(net, fg),
            "supplier_hhi": supplier_hhi(net, fg),
            "geographic_hhi": gh,
            "top_countries": dict(list(by_country.items())[:5]),
        }, indent=2))
        print("\nhidden shared sub-tier suppliers:")
        for h in hidden_dependencies(net, fg)[: args.top]:
            print(f"  {h.supplier_id:10s} depth={h.depth} tier1_branches={h.tier1_suppliers} "
                  f"parts={h.parts} revenue_at_risk={h.revenue_at_risk:14,.0f} ({h.revenue_share:.1%})")
    elif args.command == "simulate":
        print(json.dumps(simulate(net, trials=args.trials).as_dict(), indent=2))
    elif args.command == "mitigate":
        targets = [r.part_id for r in expand(net, fg) if r.sole_source and r.depth >= 2][:3]
        _, rows = score_actions(net, targets, trials=max(200, args.trials // 4))
        for r in rows[: args.top]:
            print(f"{r.action:14s} {r.target:10s} cost={r.annual_cost:12,.0f} "
                  f"reduction={r.risk_reduction:14,.0f} per_$={r.reduction_per_dollar:7.2f}")
    elif args.command == "figure":
        print(spof_figure(net, f"{args.outdir}/network_spof.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
