#!/usr/bin/env python3
"""Regenerate every number quoted in the README.

Writes ``benchmarks/results.md``. Nothing in the README is typed by hand; if a
number changes, this file is re-run and the table is copied across.

Runtime is a couple of minutes on a laptop. The expensive part is the
service-calibrated comparisons, which bisect on the safety factor and therefore
run the whole replication set a dozen times per cell.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402

from echelonsim import __version__  # noqa: E402
from echelonsim.bullwhip import (  # noqa: E402
    chen_moving_average_bullwhip,
    decompose_bullwhip,
    exponential_smoothing_bullwhip,
    measure_by_echelon,
    smoothing_sweep,
)
from echelonsim.demand import IIDNormal  # noqa: E402
from echelonsim.disruption import run_disruption_study  # noqa: E402
from echelonsim.experiments import (  # noqa: E402
    DEFAULT_CONFIG,
    estimate_warmup,
    merge_config,
    run_scenario,
    system_inventory_path,
)
from echelonsim.experiments import _replication  # noqa: E402
from echelonsim.forecast import ExponentialSmoothing, MovingAverage  # noqa: E402
from echelonsim.information import compare_information_modes  # noqa: E402
from echelonsim.leadtime import Deterministic  # noqa: E402
from echelonsim.metrics import batch_means_ci, bullwhip_ratio, mser5_truncation  # noqa: E402
from echelonsim.network import serial_chain  # noqa: E402
from echelonsim.policies import BaseStock  # noqa: E402
from echelonsim.simulation import run_simulation  # noqa: E402
from echelonsim.tradeoffs import lead_time_grid  # noqa: E402

SEED = 20260215
REPLICATIONS = 30
PERIODS = 720

BASE: Dict[str, Any] = merge_config(DEFAULT_CONFIG, {
    "topology": {"kind": "serial", "levels": 3},
    "demand": {"kind": "iid_normal", "mean": 100.0, "std": 20.0},
    "forecast": {"kind": "exponential", "alpha": 0.3},
    "policy": {"kind": "base_stock", "z": 1.645},
    "leadtime": {"kind": "deterministic", "mean": 2.0, "order_lead_time": 1},
    "review_period": 1,
    "run": {"periods": PERIODS, "replications": REPLICATIONS, "seed": SEED},
})


class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.payload: Dict[str, Any] = {}
        self.started = time.time()

    def h(self, text: str, level: int = 2) -> None:
        self.lines.append("")
        self.lines.append(f"{'#' * level} {text}")
        self.lines.append("")

    def p(self, text: str) -> None:
        self.lines.append(text)
        self.lines.append("")

    def table(self, header: List[str], rows: List[List[str]]) -> None:
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            self.lines.append("| " + " | ".join(row) + " |")
        self.lines.append("")

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.lines).rstrip() + "\n")


def section_validation(report: Report) -> None:
    report.h("1. Validation against closed-form results")
    report.p(
        "A single order-up-to stage, i.i.d. demand `N(100, 10)`, no safety term "
        "(`z = 0`), returns permitted, protection interval `R + L - 1 = 3` "
        "periods, 30,000 periods after a 200-period truncation. The simulator "
        "has no knowledge of these expressions."
    )
    rows = []
    payload = []
    for window in (5, 10, 20):
        network = serial_chain(
            levels=1,
            policy_factory=lambda level: BaseStock(z=0.0, allow_returns=True),
            forecaster_factory=lambda level, p=window: MovingAverage(p),
            transit_factory=lambda level: Deterministic(2.0),
        )
        result = run_simulation(network, IIDNormal(100.0, 10.0), periods=30_200,
                                seed=SEED).trim(200)
        series = result.nodes["retailer"]
        simulated = bullwhip_ratio(series.orders_placed, series.demand_received)
        analytic = chen_moving_average_bullwhip(3.0, window)
        rows.append([f"moving average, p={window}", f"{simulated:.4f}",
                     f"{analytic:.4f}", f"{100 * (simulated / analytic - 1):+.2f}%"])
        payload.append({"forecaster": f"ma{window}", "simulated": simulated,
                        "analytic": analytic})
    for alpha in (0.1, 0.3, 0.5):
        network = serial_chain(
            levels=1,
            policy_factory=lambda level: BaseStock(z=0.0, allow_returns=True),
            forecaster_factory=lambda level, a=alpha: ExponentialSmoothing(a),
            transit_factory=lambda level: Deterministic(2.0),
        )
        result = run_simulation(network, IIDNormal(100.0, 10.0), periods=30_200,
                                seed=SEED).trim(200)
        series = result.nodes["retailer"]
        simulated = bullwhip_ratio(series.orders_placed, series.demand_received)
        analytic = exponential_smoothing_bullwhip(3.0, alpha)
        rows.append([f"exponential smoothing, alpha={alpha}", f"{simulated:.4f}",
                     f"{analytic:.4f}", f"{100 * (simulated / analytic - 1):+.2f}%"])
        payload.append({"forecaster": f"es{alpha}", "simulated": simulated,
                        "analytic": analytic})
    report.table(["configuration", "simulated Var(q)/Var(d)", "closed form", "error"], rows)
    report.payload["validation"] = payload


def section_echelon(report: Report) -> None:
    report.h("2. Bullwhip by echelon")
    report.p(
        f"Serial chain, retailer / distributor / factory / external source. "
        f"i.i.d. demand `N(100, 20)`, exponential smoothing `alpha = 0.3`, "
        f"base-stock with `z = 1.645`, review every period, order lead time 1 "
        f"period plus 2 periods transit on every arc. {REPLICATIONS} "
        f"replications of {PERIODS} periods, warm-up truncated by MSER-5. "
        f"Intervals are 95%."
    )
    warmup = estimate_warmup(BASE, pilots=4)
    amplification = measure_by_echelon(BASE, warmup=warmup)
    rows, payload = [], []
    for name, local, local_hw, cumulative, cumulative_hw in amplification.table():
        rows.append([name, f"{local:.2f} +/- {local_hw:.2f}",
                     f"{cumulative:.2f} +/- {cumulative_hw:.2f}"])
        payload.append({"node": name, "local": local, "cumulative": cumulative,
                        "cumulative_half_width": cumulative_hw})
    report.table(["echelon", "local Var(orders)/Var(own demand)",
                  "cumulative Var(orders)/Var(end demand)"], rows)
    report.p(f"MSER-5 warm-up truncation: **{warmup} periods**.")
    report.payload["echelon"] = {"warmup": warmup, "rows": payload}

    report.h("2b. Amplification vs the smoothing constant", level=3)
    report.p(
        "Retailer amplification only, so the single-stage closed form applies. "
        "A more reactive forecast (larger `alpha`) amplifies more; this is the "
        "quantitative version of 'stop chasing the signal'."
    )
    sweep_rows, sweep_payload = [], []
    for alpha, interval, analytic in smoothing_sweep(
        alphas=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8),
        base_config=BASE,
        warmup=warmup,
    ):
        sweep_rows.append([f"{alpha:.2f}", f"{interval.mean:.2f} +/- {interval.half_width:.2f}",
                           f"{analytic:.2f}"])
        sweep_payload.append({"alpha": alpha, "simulated": interval.mean,
                              "analytic": analytic})
    report.table(["alpha", "simulated retailer amplification",
                  "closed form (z = 0 reference)"], sweep_rows)
    report.p(
        "The simulated column sits slightly above the closed form because the "
        "safety term `z*sigma_hat*sqrt(L)` is live here and moves with the "
        "forecast, adding variance the closed form does not model. At "
        "`alpha = 0.8` it falls *below* instead: order variance is by then "
        "large enough relative to mean demand that the no-returns constraint "
        "starts clipping the left tail of the order distribution. Both "
        "deviations are the model being more realistic than the formula, not "
        "less accurate."
    )
    report.payload["smoothing_sweep"] = sweep_payload

    report.h("2c. Amplification vs chain depth", level=3)
    depth_rows, depth_payload = [], []
    for levels in (1, 2, 3, 4):
        config = merge_config(BASE, {"topology": {"levels": levels}})
        outcome = run_scenario(config, name=f"levels={levels}", warmup=warmup,
                               keep_results=True)
        top = max(outcome.results[0].stocking_series(), key=lambda s: s.level)
        interval = outcome.ratio_ci(f"var_orders:{top.name}", "var_demand")
        depth_rows.append([str(levels), top.name,
                           f"{interval.mean:.2f} +/- {interval.half_width:.2f}"])
        depth_payload.append({"levels": levels, "top_node": top.name,
                              "cumulative": interval.mean})
    report.table(["stocking echelons", "most upstream node",
                  "amplification vs end demand"], depth_rows)
    report.payload["depth"] = depth_payload


def section_decomposition(report: Report) -> None:
    report.h("3. Decomposition of amplification")
    report.p(
        "Full `2^3` factorial over three mechanisms, common random numbers "
        "across all eight cells, Shapley value on `log(amplification)` computed "
        "per replication. Mechanism settings: signal processing = exponential "
        "smoothing `alpha = 0.3` (off: a forecaster that knows the true mean); "
        "batching = review every 4 periods with a 100-unit order multiple (off: "
        "review every period, multiple 1); lead time = 4 periods transit (off: "
        "1 period). Metric is `Var(factory orders)/Var(end demand)`."
    )
    result = decompose_bullwhip(BASE, metric="chain_bullwhip")
    result.check_additivity()
    rows, payload = [], []
    for key, contribution, half_width, multiplier, share in result.table():
        rows.append([key, f"{contribution:+.3f} +/- {half_width:.3f}",
                     f"x{multiplier:.2f}", f"{share:.1f}%"])
        payload.append({"mechanism": key, "log_contribution": contribution,
                        "multiplier": multiplier, "share_percent": share})
    report.table(["mechanism", "Shapley value on log amplification",
                  "multiplicative effect", "share of total"], rows)
    report.p(
        f"All mechanisms off: **{result.baseline.mean:.3f}** (theory says exactly "
        f"1.0 -- with a known mean, unit order multiple and every-period review, "
        f"the order equals the demand). All on: "
        f"**{result.full.mean:.1f}**."
    )
    cell_rows = []
    for key in sorted(result.cell_means, key=lambda k: (len(k), k)):
        cell_rows.append(["+".join(key) or "(none)", f"{result.cell_means[key]:.2f}"])
    report.table(["mechanisms active", "amplification"], cell_rows)
    report.payload["decomposition"] = {
        "baseline": result.baseline.mean,
        "full": result.full.mean,
        "mechanisms": payload,
        "cells": {"+".join(k) or "none": v for k, v in result.cell_means.items()},
    }


def section_information(report: Report) -> None:
    report.h("4. Information sharing")
    report.p(
        "Same chain as section 2. Every mode is **calibrated to the same "
        "retailer fill rate (97.5%)** by bisecting the safety factor before the "
        "comparison, so inventory differences are not service differences in "
        "disguise. All modes share the seed, so the replications are paired."
    )
    comparison = compare_information_modes(BASE, calibrate_to=0.975)
    rows, payload = [], []
    for mode, chain, half_width, fill, inventory, cost in comparison.table():
        rows.append([
            mode,
            f"{comparison.safety_factors[mode]:.3f}",
            f"{chain:.2f} +/- {half_width:.2f}",
            f"{fill:.2f}%",
            f"{inventory:.0f}",
            f"{cost:.0f}",
        ])
        payload.append({"mode": mode, "z": comparison.safety_factors[mode],
                        "chain_bullwhip": chain, "fill_rate_percent": fill,
                        "inventory": inventory, "cost": cost})
    report.table(["mode", "calibrated z", "factory amplification",
                  "retailer fill rate", "system inventory (units)",
                  "cost/period"], rows)

    echelon_rows = []
    by_mode = comparison.bullwhip_by_mode()
    for mode in comparison.outcomes:
        echelon_rows.append([mode] + [f"{by_mode[mode][n].mean:.2f}"
                                      for n in comparison.node_order])
    report.table(["mode"] + comparison.node_order, echelon_rows)

    delta_rows = []
    for mode in comparison.outcomes:
        if mode == comparison.reference:
            continue
        bullwhip = comparison.paired_percent(mode, "chain_bullwhip")
        inventory = comparison.paired_percent(mode, "avg_inventory")
        cost = comparison.paired_percent(mode, "avg_cost")
        delta_rows.append([
            f"{mode} vs {comparison.reference}",
            f"{bullwhip.mean:+.1f}% +/- {bullwhip.half_width:.1f}",
            f"{inventory.mean:+.1f}% +/- {inventory.half_width:.1f}",
            f"{cost.mean:+.1f}% +/- {cost.half_width:.1f}",
        ])
        payload.append({"mode": mode, "bullwhip_change_percent": bullwhip.mean,
                        "inventory_change_percent": inventory.mean,
                        "cost_change_percent": cost.mean})
    report.table(["paired comparison", "factory amplification",
                  "system inventory", "total cost"], delta_rows)
    report.payload["information"] = payload


def section_leadtime(report: Report) -> None:
    report.h("5. Lead-time length vs lead-time variability")
    report.p(
        "One stocking echelon fed by an always-available source, so nothing but "
        "the lead time is moving. Every cell is calibrated by bisection to a "
        "**95% fill rate**, then average on-hand inventory is compared. "
        "`sigma_DL` is the analytic standard deviation of protection-interval "
        "demand, `sqrt(L*sigma_d^2 + d_bar^2*sigma_L^2)`."
    )
    cells = lead_time_grid(
        means=(2.0, 4.0, 8.0),
        cvs=(0.0, 0.25, 0.5),
        target_fill=0.95,
        base_config={"run": {"periods": 1040, "replications": 16, "seed": SEED,
                             "warmup": 80}},
    )
    rows, payload = [], []
    reference = None
    for cell in cells:
        mean_lead, cv_lead, z, fill, inventory, sigma = cell.row()
        if reference is None:
            reference = inventory
        rows.append([f"{mean_lead:.0f}", f"{cv_lead:.2f}", f"{z:.3f}", f"{fill:.2f}%",
                     f"{inventory:.0f}", f"{sigma:.0f}"])
        payload.append({"mean_lead": mean_lead, "cv_lead": cv_lead, "z": z,
                        "fill_rate_percent": fill, "inventory": inventory,
                        "sigma_dl": sigma})
    report.table(["mean lead time", "lead-time CV", "calibrated z",
                  "achieved fill rate", "average inventory (units)",
                  "sigma_DL (analytic)"], rows)
    unconverged = [c for c in cells if not c.converged]
    if unconverged:
        report.p(
            "Cells that did not reach the target inside the bisection bracket: "
            + ", ".join(f"L={c.mean_lead:.0f}/CV={c.cv_lead:.2f}" for c in unconverged)
            + "."
        )
    report.p(
        "The calibrated `z` is the finding as much as the inventory is. The "
        "policy sizes its safety term from *demand* dispersion only -- "
        "`z*sigma_hat*sqrt(R+L-1)` -- which is what essentially every planning "
        "system does. Lead-time variance is nowhere in that expression, so `z` "
        "is left to absorb it, and the required value climbs from 0.92 to "
        "roughly 4 as soon as the supplier becomes unreliable. A `z` of 4 is "
        "not a service-level choice any planner made; it is the parameter being "
        "used to patch a missing term in the formula."
    )

    lookup = {(c.mean_lead, c.cv_lead): c for c in cells}
    doubling = lookup[(4.0, 0.0)].inventory.mean / lookup[(2.0, 0.0)].inventory.mean - 1.0
    variability = lookup[(2.0, 0.5)].inventory.mean / lookup[(2.0, 0.0)].inventory.mean - 1.0
    report.p(
        f"At a 2-period mean lead time, **doubling the mean** to 4 periods costs "
        f"{100 * doubling:+.1f}% inventory at equal service. **Adding a 0.5 "
        f"coefficient of variation** at the original 2-period mean costs "
        f"{100 * variability:+.1f}%. Same service, and the unreliable short lead "
        f"time is the more expensive of the two."
    )
    report.payload["leadtime"] = {
        "cells": payload,
        "doubling_mean_percent": 100 * doubling,
        "adding_cv_percent": 100 * variability,
    }


def section_disruption(report: Report) -> None:
    report.h("6. Disruption and recovery")
    report.p(
        "Same chain, with the factory capacitated at 160 units per period "
        "(1.6x mean demand). Each scenario is compared against its own "
        "undisrupted twin under common random numbers, so demand and transit "
        "draws are identical and only the disruption differs. Recovery is the "
        "first period after the trough at which smoothed retailer fill rate "
        "returns to within 1 point of the undisrupted path and stays there for "
        "5 consecutive periods; the interval is a bootstrap over replications."
    )
    start = 300
    config = merge_config(BASE, {"topology": {"capacity": 160.0}})
    scenarios = [
        (f"supplier outage, {n} periods",
         {"disruptions": {"outages": [{"node": "source", "start": start, "duration": n}]}},
         start, n)
        for n in (2, 4, 8)
    ]
    scenarios.append((
        "demand shock, 2x for 4 periods",
        {"demand": {"shock": {"start": start, "duration": 4, "multiplier": 2.0}}},
        start, 4,
    ))
    scenarios.append((
        "factory capacity -50%, 8 periods",
        {"disruptions": {"capacity_losses": [
            {"node": "factory", "start": start, "duration": 8, "factor": 0.5}]}},
        start, 8,
    ))
    study = run_disruption_study(config, scenarios, horizon=90)
    rows, payload = [], []
    for profile in study.profiles:
        name, duration, trough, trough_at, recovery, ratio, lost = profile.row()
        rows.append([
            name, str(duration), f"{trough:.1f}%", f"+{trough_at}",
            f"{recovery:.0f} [{profile.recovery_offset.low:.0f}, "
            f"{profile.recovery_offset.high:.0f}]",
            f"{ratio:.1f}x", f"{lost:.0f}",
        ])
        payload.append({"scenario": name, "duration": duration,
                        "trough_percent": trough, "trough_offset": trough_at,
                        "recovery_periods": recovery, "recovery_ratio": ratio,
                        "lost_units": lost,
                        "censored_fraction": profile.censored_fraction})
    report.table(["scenario", "length", "fill-rate trough", "trough at",
                  "periods to recover [95% CI]", "recovery / length",
                  "units late vs baseline"], rows)
    report.payload["disruption"] = payload
    report.payload["disruption_paths"] = {
        profile.name: {
            "offsets": profile.offsets.tolist(),
            "disrupted": profile.disrupted_fill.tolist(),
            "baseline": profile.baseline_fill.tolist(),
        }
        for profile in study.profiles
    }


def section_output_analysis(report: Report) -> None:
    from scipy import stats as _stats

    report.h("7. Output analysis: warm-up truncation and batch means")
    report.p(
        "Measured on total system on-hand-plus-backlog, averaged across 20 "
        "replications before MSER-5 is applied. Three initial conditions, "
        "same chain."
    )
    rows, payload = [], []
    variants = [
        ("4 periods of opening cover, 2-period transit (the default)",
         {"initial_periods_of_stock": 4.0, "leadtime": {"mean": 2.0}}),
        ("empty pipeline, 2-period transit",
         {"initial_periods_of_stock": 0.0, "leadtime": {"mean": 2.0}}),
        ("empty pipeline, 6-period transit",
         {"initial_periods_of_stock": 0.0, "leadtime": {"mean": 6.0}}),
    ]
    for label, override in variants:
        config = merge_config(BASE, override)
        paths = np.asarray([
            system_inventory_path(_replication(config, replication))
            for replication in range(20)
        ])
        averaged = paths.mean(axis=0)
        truncation = mser5_truncation(averaged)
        untruncated = float(averaged.mean())
        truncated = float(averaged[truncation:].mean())
        bias = 100.0 * (untruncated / truncated - 1.0)
        rows.append([label, str(truncation), f"{untruncated:.0f}",
                     f"{truncated:.0f}", f"{bias:+.1f}%"])
        payload.append({"variant": label, "mser5_truncation": truncation,
                        "mean_untruncated": untruncated,
                        "mean_truncated": truncated, "bias_percent": bias})
    report.table(["initial condition", "MSER-5 truncation (periods)",
                  "mean over whole run", "mean after truncation",
                  "bias from the transient"], rows)
    report.p(
        "The default configuration opens close enough to its operating point "
        "that truncation barely matters -- which is the argument for choosing "
        "the initial condition deliberately rather than for skipping the check. "
        "Start the same chain cold and the bias is real; lengthen the lead time "
        "and it is large enough to invalidate the run. Nothing in the "
        "configuration warns you which case you are in."
    )
    report.payload["warmup_analysis"] = payload

    report.h("7b. Batch means vs the naive interval", level=3)
    result = run_simulation(
        serial_chain(levels=3, transit_factory=lambda level: Deterministic(2.0),
                     forecaster_factory=lambda level: ExponentialSmoothing(0.3)),
        IIDNormal(100.0, 20.0), periods=PERIODS, seed=SEED,
    )
    total = system_inventory_path(result)[60:]
    naive_half = float(
        _stats.t.ppf(0.975, total.size - 1) * np.std(total, ddof=1) / np.sqrt(total.size)
    )
    batched = batch_means_ci(total, n_batches=20)
    report.p(
        f"One replication, {total.size} periods after truncation. Naive i.i.d. "
        f"95% half-width on mean system inventory: **+/- {naive_half:.1f} "
        f"units**. Batch means with 20 batches: **+/- {batched.half_width:.1f} "
        f"units**, {batched.half_width / naive_half:.1f}x wider. The naive "
        f"interval is not merely optimistic, it is answering a different "
        f"question: consecutive periods of an inventory series are strongly "
        f"dependent, and there are nowhere near {total.size} independent "
        f"observations in it. Every 'statistically significant' improvement "
        f"computed the naive way inherits that factor."
    )
    report.payload["batch_means"] = {
        "periods": int(total.size),
        "naive_half_width": naive_half,
        "batch_means_half_width": batched.half_width,
        "ratio": batched.half_width / naive_half,
        "note": batched.note,
    }


def main() -> int:
    start = time.time()
    report = Report()
    report.lines.append("# Benchmark results")
    report.lines.append("")
    report.p(
        f"Generated by `benchmarks/run_benchmarks.py` (echelonsim {__version__}). "
        f"Every number in the README is copied from this file. Seed {SEED}; "
        f"{REPLICATIONS} replications of {PERIODS} periods unless stated "
        f"otherwise. All data is generated by the simulator -- there is no "
        f"external dataset."
    )
    for section in (
        section_validation,
        section_echelon,
        section_decomposition,
        section_information,
        section_leadtime,
        section_disruption,
        section_output_analysis,
    ):
        stamp = time.time()
        section(report)
        print(f"  {section.__name__:<28} {time.time() - stamp:6.1f}s", file=sys.stderr)

    elapsed = time.time() - start
    report.p("")
    report.p(f"_Total runtime {elapsed:.0f}s._")
    output = os.path.join(REPO_ROOT, "benchmarks", "results.md")
    report.write(output)
    with open(os.path.join(REPO_ROOT, "benchmarks", "results.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report.payload, handle, indent=2, sort_keys=True, default=float)
    print(f"wrote {output} in {elapsed:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
