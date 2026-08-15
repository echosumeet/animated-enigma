"""Regenerate every number quoted in the README, and write benchmarks/results.md.

Run from the repository root:

    PYTHONPATH=src python benchmarks/run_benchmarks.py

Everything is seeded, so two runs on the same machine produce identical output.
The synthetic data-generating processes are documented inline and in the README;
no benchmark reads an external file.
"""

from __future__ import annotations

import io
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invkit.distributions import GammaLTD, MixtureLTD  # noqa: E402
from invkit.frontier import exchange_curve, marginal_cost_of_service  # noqa: E402
from invkit.guaranteed_service import example_bom_tree, solve_guaranteed_service  # noqa: E402
from invkit.leadtime import (  # noqa: E402
    LeadTimeSpec,
    lead_time_variance_share,
    ltd_moments,
    ltd_stochastic_exact,
    ltd_with_undershoot,
    undershoot_moments,
)
from invkit.lotsizing import compare_lot_sizing, seasonal_demand_series  # noqa: E402
from invkit.policies import SQPolicy, build_RS  # noqa: E402
from invkit.pooling import simulate_pooling, square_root_law  # noqa: E402
from invkit.safety_stock import (  # noqa: E402
    compare_service_definitions,
    ss_from_cycle_service_level,
    ss_from_empirical_quantile,
    ss_from_fill_rate,
)
from invkit.serial import SerialStage, clark_scarf, simulate_serial_system  # noqa: E402
from invkit.simulation import DemandProcess, simulate_policy  # noqa: E402

# ---------------------------------------------------------------------------
# The reference item.  One SKU, daily buckets.
#
#   demand      ~ Gamma with mean 100 units/day, sd 30 units/day (CV 0.30)
#   lead time   = 5 days, deterministic unless stated otherwise
#   unit cost   = 25 currency units, holding rate 25% per year
#   order cost  = 250 per replenishment
#
# Gamma is used rather than normal because it is non-negative and exactly closed
# under convolution at fixed scale, so lead-time aggregation introduces no error
# and simulated service can be compared to analytic service without an alibi.
# ---------------------------------------------------------------------------
DEMAND_MEAN = 100.0
DEMAND_SD = 30.0
LEAD_TIME = 5
UNIT_COST = 25.0
HOLDING_RATE_ANNUAL = 0.25
HOLDING_PER_DAY = UNIT_COST * HOLDING_RATE_ANNUAL / 365.0
ORDER_COST = 250.0
SIM_PERIODS = 200_000
SIM_WARMUP = 2_000

OUT = io.StringIO()


def emit(line: str = "") -> None:
    print(line)
    OUT.write(line + "\n")


def table(headers: list[str], rows: list[list[str]]) -> None:
    emit("| " + " | ".join(headers) + " |")
    emit("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        emit("| " + " | ".join(r) + " |")
    emit()


def section(title: str) -> None:
    emit()
    emit(f"## {title}")
    emit()


def base_spec(lead_time: int = LEAD_TIME) -> LeadTimeSpec:
    return LeadTimeSpec.deterministic(DEMAND_MEAN, DEMAND_SD, lead_time)


def base_process() -> DemandProcess:
    return DemandProcess(DEMAND_MEAN, DEMAND_SD, "gamma")


# ---------------------------------------------------------------------------


def bench_service_definitions() -> None:
    section("1. Cycle service level vs fill rate: the same target, two answers")
    emit(
        "Reference item, undershoot-adjusted lead-time demand "
        f"(mean {ltd_with_undershoot(base_spec(), DEMAND_MEAN, DEMAND_SD).mean:,.1f}, "
        f"sd {ltd_with_undershoot(base_spec(), DEMAND_MEAN, DEMAND_SD).sd:,.1f}). "
        "`Q/sigma` is the order quantity in standard deviations of lead-time demand - "
        "the single number that determines how far the two definitions diverge."
    )
    emit()
    ltd = ltd_with_undershoot(base_spec(), DEMAND_MEAN, DEMAND_SD)
    rows = []
    for Q in (200.0, 400.0, 800.0, 1600.0, 3200.0):
        for target in (0.95, 0.98):
            c = compare_service_definitions(ltd, target, Q)
            rows.append(
                [
                    f"{target:.2f}",
                    f"{Q:,.0f}",
                    f"{c['Q_over_sigma']:.1f}",
                    f"{c['ss_csl_basis']:,.0f}",
                    f"{c['ss_fill_basis']:,.0f}",
                    f"{c['ss_delta']:,.0f}",
                    f"{c['fill_at_csl_basis']:.4f}",
                    f"{c['csl_at_fill_basis']:.3f}",
                ]
            )
    table(
        [
            "target",
            "Q",
            "Q/sigma",
            "SS if read as CSL",
            "SS if read as fill",
            "difference",
            "fill achieved by CSL sizing",
            "CSL achieved by fill sizing",
        ],
        rows,
    )
    c = compare_service_definitions(ltd, 0.95, 800.0)
    emit(
        f"At the reference lot of 800 units, reading a 95% target as cycle service holds "
        f"{c['ss_delta']:,.0f} more units of safety stock than reading it as fill rate, and "
        f"delivers a fill rate of {c['fill_at_csl_basis']:.4f} instead of 0.95. "
        f"At {UNIT_COST:,.0f} per unit that is "
        f"{c['ss_delta'] * UNIT_COST:,.0f} of working capital, per SKU, "
        "bought by a definition rather than by a decision."
    )


def bench_simulation_validation() -> None:
    section("2. Analytic service vs simulated service")
    emit(
        f"Each policy is built from a closed-form target and then simulated for "
        f"{SIM_PERIODS:,} days ({SIM_WARMUP:,} warm-up). Cycle service is measured per "
        "replenishment cycle; fill rate is units served from stock over units demanded."
    )
    emit()
    proc = base_process()
    ltd = ltd_with_undershoot(base_spec(), DEMAND_MEAN, DEMAND_SD)
    Q = 800.0
    rows = []
    for target in (0.90, 0.95, 0.98):
        res = ss_from_cycle_service_level(ltd, target, Q)
        sim = simulate_policy(
            SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
            n_periods=SIM_PERIODS, warmup=SIM_WARMUP, seed=101,
        )
        rows.append(
            [
                "(s,Q) cycle service",
                f"{target:.3f}",
                f"{res.reorder_point:,.0f}",
                f"{sim.cycle_service_level:.4f}",
                f"{sim.cycle_service_level - target:+.4f}",
                f"{sim.n_cycles:,}",
            ]
        )
    for target in (0.95, 0.98, 0.995):
        res = ss_from_fill_rate(ltd, target, Q)
        sim = simulate_policy(
            SQPolicy(s=res.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
            n_periods=SIM_PERIODS, warmup=SIM_WARMUP, seed=303,
        )
        rows.append(
            [
                "(s,Q) fill rate",
                f"{target:.3f}",
                f"{res.reorder_point:,.0f}",
                f"{sim.fill_rate:.4f}",
                f"{sim.fill_rate - target:+.4f}",
                f"{sim.n_cycles:,}",
            ]
        )
    for target in (0.95, 0.98):
        policy, res = build_RS(base_spec(3), R=1, target_csl=target)
        sim = simulate_policy(
            policy, proc, {3: 1.0}, n_periods=SIM_PERIODS, warmup=SIM_WARMUP, seed=505
        )
        rows.append(
            [
                "(R,S) R=1 ready rate",
                f"{target:.3f}",
                f"{policy.S:,.0f}",
                f"{sim.ready_rate:.4f}",
                f"{sim.ready_rate - target:+.4f}",
                f"{sim.periods:,}",
            ]
        )
    for target in (0.95, 0.98):
        policy, res = build_RS(base_spec(3), R=5, target_fill=target)
        sim = simulate_policy(
            policy, proc, {3: 1.0}, n_periods=SIM_PERIODS, warmup=SIM_WARMUP, seed=606
        )
        rows.append(
            [
                "(R,S) R=5 fill rate",
                f"{target:.3f}",
                f"{policy.S:,.0f}",
                f"{sim.fill_rate:.4f}",
                f"{sim.fill_rate - target:+.4f}",
                f"{sim.periods:,}",
            ]
        )
    table(["policy / measure", "target", "s or S", "simulated", "error", "cycles / periods"], rows)

    naive = ss_from_cycle_service_level(
        GammaLTD.from_moments(*ltd_moments(base_spec())), 0.95, Q
    )
    sim_naive = simulate_policy(
        SQPolicy(s=naive.reorder_point, Q=Q), proc, {LEAD_TIME: 1.0},
        n_periods=SIM_PERIODS, warmup=SIM_WARMUP, seed=101,
    )
    eu, vu = undershoot_moments(DEMAND_MEAN, DEMAND_SD)
    emit(
        f"Control: drop the undershoot correction (E[U] = {eu:.1f} units) and the same 95% "
        f"cycle-service target delivers {sim_naive.cycle_service_level:.4f} in simulation - "
        f"{100 * (0.95 - sim_naive.cycle_service_level):.1f} points short. That is the "
        "uncorrected textbook formula, and it is what most planning systems run."
    )


def bench_empirical_tails() -> None:
    section("3. Empirical forecast-error quantiles vs the normal assumption")
    rng = np.random.default_rng(20260815)
    n = 100_000
    base = rng.lognormal(0.0, 0.35, int(0.9 * n))
    spike = rng.lognormal(1.0, 0.55, n - int(0.9 * n))
    raw = np.concatenate([base, spike])
    rng.shuffle(raw)
    errors = (raw - raw.mean()) / raw.std(ddof=1) * DEMAND_SD
    skew = float(((errors - errors.mean()) ** 3).mean() / errors.std() ** 3)
    kurt = float(((errors - errors.mean()) ** 4).mean() / errors.std() ** 4)
    emit(
        "Data-generating process: a 90/10 mixture of a tight lognormal and a "
        "promotion-driven spike, rescaled to zero mean and the reference "
        f"per-day standard deviation of {DEMAND_SD:.0f} units. "
        f"Sample skewness {skew:.2f}, kurtosis {kurt:.1f}. Errors are aggregated over the "
        "lead time by iid bootstrap; the normal comparator uses the *same* aggregated "
        "moments, so the only difference between the two columns is distribution shape."
    )
    emit()
    rows = []
    for L in (1, 5, 10, 20):
        for target in (0.95, 0.98, 0.99):
            emp, norm = ss_from_empirical_quantile(errors, target, L)
            rows.append(
                [
                    str(L),
                    f"{target:.2f}",
                    f"{emp:,.0f}",
                    f"{norm:,.0f}",
                    f"{100 * (emp / norm - 1):+.1f}%",
                ]
            )
    table(["lead time (days)", "target", "empirical SS", "normal SS", "gap"], rows)
    emp5, norm5 = ss_from_empirical_quantile(errors, 0.98, 5)
    emp20, norm20 = ss_from_empirical_quantile(errors, 0.98, 20)
    emit(
        f"At a 5-day lead time and a 98% target the normal assumption understates safety "
        f"stock by {100 * (1 - norm5 / emp5):.0f}%. The gap narrows as the lead time "
        f"lengthens ({100 * (emp20 / norm20 - 1):+.0f}% at 20 days) because summing "
        "independent errors pulls the aggregate toward normal - so short-lead-time, "
        "high-skew items are where a normal fit hurts most, which is the opposite of "
        "where most people look for the problem."
    )


def bench_stochastic_lead_time() -> None:
    section("4. Stochastic lead time: exact mixture vs variance convolution")
    pmf = {2: 0.7, 14: 0.3}
    spec = LeadTimeSpec(DEMAND_MEAN, DEMAND_SD, pmf)
    mean, sd = ltd_moments(spec)
    exact = ltd_stochastic_exact(spec, family="gamma")
    matched = GammaLTD.from_moments(mean, sd)
    emit(
        "Lead time is 2 days 70% of the time and 14 days 30% of the time - a supplier that "
        "is usually domestic and occasionally not. Both models have identical mean "
        f"({mean:,.0f}) and standard deviation ({sd:,.0f}); they disagree about shape."
    )
    emit()
    emit(
        f"Share of lead-time demand variance coming from lead-time variability: "
        f"{lead_time_variance_share(spec):.1%}."
    )
    emit()
    rows = []
    for target in (0.75, 0.85, 0.90, 0.95, 0.98, 0.99):
        e, m = exact.ppf(target), matched.ppf(target)
        rows.append(
            [f"{target:.2f}", f"{e:,.0f}", f"{m:,.0f}", f"{100 * (m / e - 1):+.0f}%"]
        )
    table(
        ["cycle service target", "reorder point (exact mixture)",
         "reorder point (moment-matched)", "error of moment matching"],
        rows,
    )
    emit(
        "Matching two moments to a unimodal shape cannot represent two modes. It "
        "understates the reorder point across the whole service range a planner would "
        "actually pick, then overstates it far out in the tail. The understatement is the "
        "dangerous one: it is invisible until the long lead time lands."
    )

    proc = base_process()
    pmf2 = {3: 0.6, 9: 0.4}
    eu, vu = undershoot_moments(DEMAND_MEAN, DEMAND_SD)
    components = [
        GammaLTD.from_moments(l * DEMAND_MEAN + eu, math.sqrt(l * DEMAND_SD ** 2 + vu))
        for l in sorted(pmf2)
    ]
    mixture = MixtureLTD(components, [pmf2[l] for l in sorted(pmf2)])
    rows = []
    for Q in (800.0, 1500.0):
        for target in (0.90, 0.95):
            res = ss_from_cycle_service_level(mixture, target, Q)
            sim = simulate_policy(
                SQPolicy(s=res.reorder_point, Q=Q), proc, pmf2,
                n_periods=SIM_PERIODS, warmup=SIM_WARMUP, seed=202,
            )
            rows.append(
                [
                    f"{Q:,.0f}",
                    f"{Q / (DEMAND_MEAN * max(pmf2)):.2f}",
                    f"{target:.2f}",
                    f"{res.reorder_point:,.0f}",
                    f"{sim.cycle_service_level:.4f}",
                    f"{sim.cycle_service_level - target:+.4f}",
                ]
            )
    emit()
    emit(
        "Validation with a bimodal lead time of 3 days (60%) or 9 days (40%). The column "
        "`Q / max lead-time demand` is the condition under which at most one order is "
        "outstanding, which is what the single-cycle formula assumes:"
    )
    emit()
    table(
        ["Q", "Q / max lead-time demand", "target", "reorder point", "simulated CSL", "error"],
        rows,
    )
    emit(
        "When the lot is smaller than the longest lead-time demand, a second order goes out "
        "before the first lands, two orders are exposed to the same demand, and realised "
        "service falls well below the formula. Long variable lead times with small lots - "
        "which describes most low-volume imported parts - sit squarely in that regime."
    )


def bench_lot_sizing() -> None:
    section("5. Lot sizing: exact dynamic program vs the heuristics MRP actually runs")
    emit(
        "50 random 18-period demand series: each period is drawn uniform on [0, 200] and "
        "rounded, with setup cost 300 and holding cost 1 per unit per period. "
        "Wagner-Whitin is exact, so every gap below is a true optimality gap."
    )
    emit()
    rng = random.Random(20260815)
    gaps: dict[str, list[float]] = {}
    setups: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for _ in range(50):
        demand = [float(rng.randint(0, 200)) for _ in range(18)]
        plans = compare_lot_sizing(demand, 300.0, 1.0)
        opt = plans["wagner-whitin"]
        for name, plan in plans.items():
            gaps.setdefault(name, []).append(100.0 * plan.gap_vs(opt))
            setups.setdefault(name, []).append(plan.n_setups)
    elapsed = time.perf_counter() - t0
    rows = []
    for name in ("wagner-whitin", "silver-meal", "least-unit-cost", "lot-for-lot"):
        g = np.asarray(gaps[name])
        rows.append(
            [
                name,
                f"{g.mean():.2f}%",
                f"{np.median(g):.2f}%",
                f"{g.max():.2f}%",
                f"{np.mean(setups[name]):.1f}",
            ]
        )
    table(["method", "mean gap", "median gap", "worst gap", "mean orders placed"], rows)
    emit(
        f"All four methods over 50 instances in {elapsed * 1000:.0f} ms. Wagner-Whitin is O(T^2) and "
        "cheap enough that there is no computational argument left for running a heuristic; "
        "the argument for Silver-Meal is that it is what the ERP already does."
    )
    emit()
    demand = seasonal_demand_series(12, 120.0, 70.0, noise_cv=0.25, seed=5)
    plans = compare_lot_sizing(demand, 300.0, 1.0)
    emit("A single seasonal instance (12 periods, sinusoidal with 25% multiplicative noise):")
    emit()
    table(
        ["method", "orders", "setup cost", "holding cost", "total", "gap"],
        [
            [
                p.method,
                str(p.n_setups),
                f"{p.setup_cost:,.0f}",
                f"{p.holding_cost:,.0f}",
                f"{p.total_cost:,.0f}",
                f"{100 * p.gap_vs(plans['wagner-whitin']):.2f}%",
            ]
            for p in plans.values()
        ],
    )


def bench_clark_scarf() -> None:
    section("6. Clark-Scarf serial system: dynamic program vs simulation")
    emit(
        "Three-stage serial system (store <- dc <- plant) with echelon holding costs "
        "1.2 / 0.5 / 0.4 per unit per period, lead times 1 / 2 / 4 periods, backorder "
        "penalty 20 per unit per period, and discretised normal demand of mean 100, sd 30. "
        "The dynamic program returns the exact optimal cost under that demand model; the "
        "simulation runs the resulting echelon base-stock policy and measures what it costs."
    )
    emit()
    configs = [
        ("2-stage", [SerialStage("retail", 2, 1.0), SerialStage("dc", 3, 0.6)]),
        (
            "3-stage",
            [
                SerialStage("store", 1, 1.2),
                SerialStage("dc", 2, 0.5),
                SerialStage("plant", 4, 0.4),
            ],
        ),
        (
            "4-stage",
            [
                SerialStage("store", 1, 1.2),
                SerialStage("dc", 2, 0.5),
                SerialStage("plant", 3, 0.4),
                SerialStage("supplier", 5, 0.3),
            ],
        ),
    ]
    rows = []
    for label, stages in configs:
        t0 = time.perf_counter()
        sol = clark_scarf(stages, DEMAND_MEAN, DEMAND_SD, penalty=20.0)
        dp_time = time.perf_counter() - t0
        sim = simulate_serial_system(
            stages, sol.echelon_base_stock, DEMAND_MEAN, DEMAND_SD, 20.0,
            n_periods=120_000, seed=17,
        )
        rows.append(
            [
                label,
                " / ".join(f"{s:,.0f}" for s in sol.echelon_base_stock),
                f"{sol.optimal_cost:,.2f}",
                f"{sim['avg_cost_per_period']:,.2f}",
                f"{100 * (sim['avg_cost_per_period'] / sol.optimal_cost - 1):+.2f}%",
                f"{sim['fill_rate']:.4f}",
                f"{dp_time * 1000:.0f} ms",
            ]
        )
    table(
        ["system", "echelon base stock (downstream first)", "DP cost/period",
         "simulated cost/period", "gap", "simulated fill rate", "DP time"],
        rows,
    )

    stages = configs[1][1]
    sol = clark_scarf(stages, DEMAND_MEAN, DEMAND_SD, penalty=20.0)
    base = simulate_serial_system(
        stages, sol.echelon_base_stock, DEMAND_MEAN, DEMAND_SD, 20.0, n_periods=120_000, seed=29
    )["avg_cost_per_period"]
    rows = []
    for label, delta in (
        ("store +60", (60, 0, 0)),
        ("store -60", (-60, 0, 0)),
        ("dc +80", (0, 80, 0)),
        ("plant +120", (0, 0, 120)),
        ("all +10%", tuple(int(0.1 * s) for s in sol.echelon_base_stock)),
    ):
        perturbed = [s + d for s, d in zip(sol.echelon_base_stock, delta)]
        cost = simulate_serial_system(
            stages, perturbed, DEMAND_MEAN, DEMAND_SD, 20.0, n_periods=120_000, seed=29
        )["avg_cost_per_period"]
        rows.append([label, f"{cost:,.2f}", f"{100 * (cost / base - 1):+.2f}%"])
    emit(
        "Perturbing the optimal echelon levels and re-simulating - the optimum is a real "
        "optimum, not a fixed point of the recursion:"
    )
    emit()
    table(["perturbation", "simulated cost/period", "vs optimal"], rows)


def bench_guaranteed_service() -> None:
    section("7. Guaranteed-service safety stock placement on a BOM tree")
    tree = example_bom_tree()
    demand = tree.propagate_demand()
    emit(
        "Nine-stage assembly network: two raw materials feed a sub-assembly; that plus a "
        "purchased part feed a build stage; build feeds pack and a service-parts channel; "
        "pack feeds two regional DCs. External demand is 120 +/- 40 units/day at the north "
        "DC, 80 +/- 35 at the south DC and 15 +/- 12 on service parts. `raw_A` is consumed "
        "2 per sub-assembly. Holding rate 25% per period of cumulative unit cost, "
        "demand-bound service level 95%."
    )
    emit()
    t0 = time.perf_counter()
    res = solve_guaranteed_service(tree, 0.95, 0.25)
    elapsed = time.perf_counter() - t0
    rows = []
    for name in sorted(tree.stages):
        st = tree.stages[name]
        rows.append(
            [
                name,
                str(st.processing_time),
                f"{st.cost_added:,.0f}",
                f"{demand[name][1]:,.1f}",
                str(res.inbound_service_times[name]),
                str(res.service_times[name]),
                str(res.net_replenishment_times[name]),
                f"{res.safety_stock[name]:,.0f}",
                f"{res.safety_stock_cost[name]:,.0f}",
            ]
        )
    table(
        ["stage", "T", "unit cost", "demand sd", "inbound service", "outbound service",
         "net repl. time", "safety stock", "cost/period"],
        rows,
    )
    zero = sorted(k for k, v in res.net_replenishment_times.items() if v == 0)
    emit(
        f"Total safety-stock holding cost {res.total_cost:,.0f} per period, solved in "
        f"{elapsed * 1000:.0f} ms. {len(zero)} of {len(tree.stages)} stages hold no safety "
        f"stock at all ({', '.join(zero)}); the buffer concentrates at "
        f"{len(res.decoupling_points)} decoupling points. That is the characteristic "
        "guaranteed-service answer, and it is a fundamentally different recommendation from "
        "'give every stage a 95% service level'."
    )
    emit()
    rows = []
    for sl in (0.90, 0.95, 0.98, 0.99):
        r = solve_guaranteed_service(tree, sl, 0.25)
        rows.append(
            [
                f"{sl:.2f}",
                f"{r.z:.3f}",
                f"{r.total_cost:,.0f}",
                str(len(r.decoupling_points)),
                ", ".join(r.decoupling_points),
            ]
        )
    emit("Placement is invariant to the service level; only the scale changes:")
    emit()
    table(["service level", "z", "total cost/period", "decoupling points", "which stages"], rows)


def bench_pooling() -> None:
    section("8. Risk pooling: the square-root law and what breaks it")
    emit(
        f"Identical locations, each with per-day demand standard deviation {DEMAND_SD:.0f}, "
        "95% service. Correlation is equicorrelated across locations. The simulated column "
        "draws 300,000 correlated normal demand vectors via a single common factor and "
        "compares the sum of per-location quantile buffers to the buffer on the pooled total."
    )
    emit()
    rows = []
    for n in (2, 4, 8, 16):
        for rho in (0.0, 0.3, 0.6):
            r = square_root_law(n, DEMAND_SD, 0.95, correlation=rho)
            sim = simulate_pooling(n, 100.0, DEMAND_SD, correlation=rho, n_draws=300_000, seed=8)
            rows.append(
                [
                    str(n),
                    f"{rho:.1f}",
                    f"{r.decentralised_ss:,.0f}",
                    f"{r.centralised_ss:,.0f}",
                    f"{r.reduction_pct:.1f}%",
                    f"{r.effective_sqrt_n:.2f}",
                    f"{sim['simulated_ratio']:.2f}",
                ]
            )
    table(
        ["locations", "correlation", "decentralised SS", "centralised SS",
         "reduction", "analytic ratio", "simulated ratio"],
        rows,
    )
    r0 = square_root_law(8, DEMAND_SD, 0.95, correlation=0.0)
    r6 = square_root_law(8, DEMAND_SD, 0.95, correlation=0.6)
    emit(
        f"Consolidating 8 independent locations cuts safety stock {r0.reduction_pct:.0f}%. "
        f"At a realistic 0.6 correlation between regions - one national promotion moves them "
        f"all - the same consolidation delivers {r6.reduction_pct:.0f}%. Network-design "
        "cases that quote the independent number are overstating the prize by roughly a "
        "factor of two before a single mile of extra outbound freight is counted."
    )


def bench_frontier() -> None:
    section("9. Cost-service efficient frontier")
    ltd = ltd_with_undershoot(base_spec(), DEMAND_MEAN, DEMAND_SD)
    curves = exchange_curve(ltd, 800.0, UNIT_COST, HOLDING_RATE_ANNUAL / 365.0, n_points=14)
    emit(
        f"Reference item, lot 800 units, unit cost {UNIT_COST:,.0f}, holding rate "
        f"{HOLDING_RATE_ANNUAL:.0%} per year. Holding cost below is per day on cycle stock "
        "plus safety stock."
    )
    emit()
    rows = []
    for f, c in zip(curves["fill"], curves["csl"]):
        rows.append(
            [
                f"{f.target:.3f}",
                f"{f.safety_stock:,.0f}",
                f"{f.holding_cost:,.3f}",
                f"{c.safety_stock:,.0f}",
                f"{c.holding_cost:,.3f}",
                f"{c.achieved_fill:.5f}",
            ]
        )
    table(
        ["target", "SS (fill basis)", "holding/day (fill basis)",
         "SS (CSL basis)", "holding/day (CSL basis)", "fill delivered by CSL basis"],
        rows,
    )
    marg = marginal_cost_of_service(curves["fill"])
    emit("Marginal cost of the next point of fill rate:")
    emit()
    table(
        ["from fill", "to fill", "extra holding cost/day", "cost per service point"],
        [
            [
                f"{r['from_fill']:.4f}",
                f"{r['to_fill']:.4f}",
                f"{r['delta_holding_cost']:.4f}",
                f"{r['cost_per_service_point']:.4f}",
            ]
            for r in marg[-6:]
        ],
    )
    first, last = marg[0], marg[-1]
    emit(
        f"The last point of fill rate on this curve costs "
        f"{last['cost_per_service_point'] / first['cost_per_service_point']:.1f}x what the "
        "first one did. That ratio, not a corporate target, is the argument for "
        "differentiating service by segment."
    )


def bench_timings() -> None:
    section("10. Runtime")
    proc = base_process()
    ltd = ltd_with_undershoot(base_spec(), DEMAND_MEAN, DEMAND_SD)
    rows = []

    t0 = time.perf_counter()
    for _ in range(1000):
        ss_from_fill_rate(ltd, 0.98, 800.0)
    rows.append(["fill-rate inversion (gamma, exact)", f"{(time.perf_counter() - t0):.3f} s / 1,000 solves"])

    t0 = time.perf_counter()
    simulate_policy(
        SQPolicy(s=ltd.ppf(0.95), Q=800.0), proc, {LEAD_TIME: 1.0},
        n_periods=100_000, warmup=1_000, seed=1,
    )
    rows.append(["policy simulation", f"{(time.perf_counter() - t0):.2f} s / 100,000 periods"])

    stages = [SerialStage("store", 1, 1.2), SerialStage("dc", 2, 0.5), SerialStage("plant", 4, 0.4)]
    t0 = time.perf_counter()
    clark_scarf(stages, DEMAND_MEAN, DEMAND_SD, penalty=20.0)
    rows.append(["Clark-Scarf DP (3 stages)", f"{(time.perf_counter() - t0) * 1000:.0f} ms"])

    tree = example_bom_tree()
    t0 = time.perf_counter()
    solve_guaranteed_service(tree, 0.95, 0.25)
    rows.append(["Graves-Willems DP (9 stages)", f"{(time.perf_counter() - t0) * 1000:.0f} ms"])

    demand = [float(random.Random(1).randint(0, 200)) for _ in range(200)]
    t0 = time.perf_counter()
    compare_lot_sizing(demand, 300.0, 1.0)
    rows.append(["Wagner-Whitin + heuristics (200 periods)", f"{(time.perf_counter() - t0) * 1000:.0f} ms"])

    table(["operation", "time"], rows)
    emit(
        f"Python {sys.version.split()[0]}, numpy {np.__version__}. Single core, no "
        "compilation step, no parallelism."
    )


def main() -> int:
    started = time.perf_counter()
    emit("# invkit benchmark results")
    emit()
    emit(
        "Generated by `python benchmarks/run_benchmarks.py`. Every number in the README "
        "comes from this file. All inputs are synthetic and generated in code; the "
        "data-generating processes are stated with each table."
    )
    bench_service_definitions()
    bench_simulation_validation()
    bench_empirical_tails()
    bench_stochastic_lead_time()
    bench_lot_sizing()
    bench_clark_scarf()
    bench_guaranteed_service()
    bench_pooling()
    bench_frontier()
    bench_timings()
    emit()
    emit(f"Total benchmark runtime: {time.perf_counter() - started:.1f} s.")
    (Path(__file__).parent / "results.md").write_text(OUT.getvalue(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
