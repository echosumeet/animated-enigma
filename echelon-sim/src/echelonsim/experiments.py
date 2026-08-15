"""Experiment configuration, replication, and comparison.

A configuration is a plain nested ``dict`` -- JSON on disk, no schema library,
no custom DSL. That is a deliberate choice: an experiment config is a document
that has to be readable in a code review and diff-able in a pull request six
months later, and the moment it becomes objects it stops being either.

The replication machinery does three things that are easy to get wrong:

1. **Common random numbers across scenarios.** Every scenario in a comparison
   is run with the same ``seed`` and the same replication indices, and each
   stochastic element draws from its own named stream. Replication 7 of
   "decentralised" and replication 7 of "VMI" therefore see identical customer
   demand and identical transit draws, and the difference between them is
   attributable to the design change rather than to luck.

2. **Warm-up truncation, estimated rather than guessed.** With ``warmup:
   "auto"`` the framework runs a small number of pilot replications, applies
   MSER-5 to the system-wide on-hand series, takes the largest truncation point
   across the pilots, and uses that for every scenario in the study. Using the
   *same* truncation everywhere matters: a per-scenario truncation would make
   the scenarios cover different stretches of simulated time, which is a subtle
   way to compare two things that are not comparable.

3. **Confidence intervals on everything.** A single simulated bullwhip ratio is
   not a result. Ratios get log-scale intervals; differences between paired
   scenarios get paired-t intervals.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .demand import AR1, DemandProcess, IIDNormal, SeasonalTrend, ShockOverlay
from .forecast import DampedTrend, ExponentialSmoothing, Forecaster, MovingAverage, Oracle
from .leadtime import Deterministic, DiscreteLeadTime, GammaLeadTime, LeadTime, NoCrossingWrapper
from .metrics import ConfidenceInterval, mean_ci, mser5_truncation, paired_difference_ci, variance_ratio_ci
from .network import InfoMode, SupplyNetwork, divergent_network, serial_chain
from .policies import Batched, BaseStock, Policy, RSPolicy, SsPolicy
from .simulation import CapacityLoss, DisruptionPlan, SimulationResult, Simulator, SupplyOutage
from .rng import StreamBank

__all__ = [
    "DEFAULT_CONFIG",
    "merge_config",
    "build_demand",
    "build_network",
    "build_disruptions",
    "ScenarioOutcome",
    "run_scenario",
    "compare_scenarios",
    "percent_reduction",
    "estimate_warmup",
    "system_inventory_path",
    "build_forecaster",
    "build_policy",
    "build_leadtime",
    "save_config",
    "load_config",
]


DEFAULT_CONFIG: Dict[str, Any] = {
    "topology": {"kind": "serial", "levels": 3, "n_retailers": 3, "capacity": None},
    "demand": {"kind": "iid_normal", "mean": 100.0, "std": 20.0, "rho": 0.6},
    "forecast": {"kind": "moving_average", "window": 10, "alpha": 0.3, "phi": 0.85},
    "policy": {"kind": "base_stock", "z": 1.645, "allow_returns": False,
               "batch_multiple": 1.0, "min_order": 0.0, "quantity_periods": 4.0},
    "leadtime": {"kind": "deterministic", "mean": 2.0, "cv": 0.0,
                 "order_lead_time": 1, "no_crossing": False},
    "review_period": 1,
    "info_mode": "decentralized",
    "initial_periods_of_stock": 4.0,
    "run": {"periods": 520, "replications": 20, "warmup": "auto", "seed": 20260215},
    "disruptions": {"outages": [], "capacity_losses": []},
}


def merge_config(base: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Recursive dict merge. ``override`` wins; lists are replaced, not appended."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return merge_config(DEFAULT_CONFIG, json.load(handle))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _at_level(value: Any, level: int) -> Any:
    """Allow any scalar parameter to be given per level as a list."""
    if isinstance(value, (list, tuple)):
        return value[min(level, len(value) - 1)]
    return value


def build_demand(config: Dict[str, Any]) -> DemandProcess:
    spec = config["demand"]
    kind = spec["kind"]
    if kind == "iid_normal":
        base: DemandProcess = IIDNormal(mean=spec["mean"], std=spec["std"])
    elif kind == "ar1":
        base = AR1(mean=spec["mean"], std=spec["std"], rho=spec["rho"])
    elif kind == "seasonal":
        base = SeasonalTrend(
            mean=spec["mean"], std=spec["std"],
            amplitude=spec.get("amplitude", 30.0),
            period_length=spec.get("period_length", 12),
            drift=spec.get("drift", 0.0),
        )
    else:
        raise ValueError(f"unknown demand kind {kind!r}")
    shock = spec.get("shock")
    if shock:
        base = ShockOverlay(
            base=base,
            start=int(shock["start"]),
            duration=int(shock["duration"]),
            multiplier=float(shock.get("multiplier", 1.0)),
            offset=float(shock.get("offset", 0.0)),
        )
    return base


def build_forecaster(config: Dict[str, Any], level: int) -> Forecaster:
    spec = config["forecast"]
    kind = _at_level(spec["kind"], level)
    if kind == "moving_average":
        return MovingAverage(window=int(_at_level(spec["window"], level)))
    if kind == "exponential":
        return ExponentialSmoothing(alpha=float(_at_level(spec["alpha"], level)))
    if kind == "damped_trend":
        return DampedTrend(
            alpha=float(_at_level(spec["alpha"], level)),
            phi=float(_at_level(spec["phi"], level)),
        )
    if kind == "oracle":
        return Oracle()
    raise ValueError(f"unknown forecast kind {kind!r}")


def build_policy(config: Dict[str, Any], level: int) -> Policy:
    spec = config["policy"]
    kind = _at_level(spec["kind"], level)
    z = float(_at_level(spec["z"], level))
    if kind == "base_stock":
        inner: Policy = BaseStock(z=z, allow_returns=bool(spec.get("allow_returns", False)))
    elif kind == "ss":
        inner = SsPolicy(z=z, quantity_periods=float(_at_level(spec["quantity_periods"], level)))
    elif kind == "rs":
        inner = RSPolicy(z=z, review_period=int(config["review_period"]),
                         allow_returns=bool(spec.get("allow_returns", False)))
    else:
        raise ValueError(f"unknown policy kind {kind!r}")
    multiple = float(_at_level(spec.get("batch_multiple", 1.0), level))
    minimum = float(_at_level(spec.get("min_order", 0.0), level))
    if multiple > 1.0 or minimum > 0.0:
        return Batched(inner=inner, multiple=multiple, minimum=minimum)
    return inner


def build_leadtime(config: Dict[str, Any], level: int) -> LeadTime:
    spec = config["leadtime"]
    kind = _at_level(spec["kind"], level)
    if kind == "deterministic":
        base: LeadTime = Deterministic(float(_at_level(spec["mean"], level)))
    elif kind == "gamma":
        cv = float(_at_level(spec["cv"], level))
        mean = float(_at_level(spec["mean"], level))
        if cv <= 0:
            base = Deterministic(mean)
        else:
            base = GammaLeadTime(mean_lt=mean, cv_lt=cv,
                                 minimum=float(spec.get("minimum", 0.5)))
    elif kind == "discrete":
        base = DiscreteLeadTime(values=spec["values"], probabilities=spec["probabilities"])
    else:
        raise ValueError(f"unknown lead time kind {kind!r}")
    if spec.get("no_crossing", False) and base.std > 0:
        return NoCrossingWrapper(base=base)
    return base


def build_network(config: Dict[str, Any]) -> SupplyNetwork:
    topology = config["topology"]
    info_mode = InfoMode(config["info_mode"])
    review = int(config["review_period"])
    kwargs = dict(
        policy_factory=lambda level: build_policy(config, level),
        forecaster_factory=lambda level: build_forecaster(config, level),
        transit_factory=lambda level: build_leadtime(config, level),
        review_period=review,
        order_lead_time=int(config["leadtime"]["order_lead_time"]),
        initial_periods_of_stock=float(config["initial_periods_of_stock"]),
        info_mode=info_mode,
    )
    if topology["kind"] == "serial":
        network = serial_chain(levels=int(topology["levels"]),
                               capacity=topology.get("capacity"), **kwargs)
    elif topology["kind"] == "divergent":
        network = divergent_network(n_retailers=int(topology["n_retailers"]),
                                    factory_capacity=topology.get("capacity"), **kwargs)
    else:
        raise ValueError(f"unknown topology {topology['kind']!r}")
    if info_mode is InfoMode.VMI:
        # Vendor-managed replenishment is continuous at the point of sale: the
        # vendor is looking at the shelf, not waiting for a review cycle.
        for retailer in network.retailers:
            retailer.review_period = 1
    return network


def build_disruptions(config: Dict[str, Any]) -> DisruptionPlan:
    spec = config.get("disruptions") or {}
    outages = tuple(
        SupplyOutage(node=item["node"], start=int(item["start"]), duration=int(item["duration"]))
        for item in spec.get("outages", [])
    )
    losses = tuple(
        CapacityLoss(node=item["node"], start=int(item["start"]),
                     duration=int(item["duration"]), factor=float(item.get("factor", 0.5)))
        for item in spec.get("capacity_losses", [])
    )
    return DisruptionPlan(outages=outages, capacity_losses=losses)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _replication(config: Dict[str, Any], replication: int) -> SimulationResult:
    network = build_network(config)
    demand = build_demand(config)
    plan = build_disruptions(config)
    streams = StreamBank(seed=int(config["run"]["seed"]), replication=replication)
    return Simulator(network, demand, int(config["run"]["periods"]), streams, plan).run()


def system_inventory_path(result: SimulationResult) -> np.ndarray:
    """Total on-hand plus backlog across every stocking node, per period."""
    total = np.zeros(result.periods, dtype=float)
    for series in result.stocking_series():
        total += series.on_hand + series.backlog
    return total


def estimate_warmup(config: Dict[str, Any], pilots: int = 4) -> int:
    """MSER-5 truncation point from pilot replications of ``config``.

    The statistic is applied to total system on-hand-plus-backlog, which carries
    the longest transient in this model: the pipeline has to fill before any
    node reaches its operating point, and that takes roughly the *sum* of the
    lead times down the chain, not the maximum.

    MSER-5 is applied to the **replication-averaged** path rather than to each
    pilot separately. On a single run the statistic is noisy enough to return
    zero on a series with an obvious transient -- it is trading a genuine
    variance reduction against the ``1/(n-d)`` penalty for discarding data, and
    on one path the noise wins. Averaging first collapses that noise by
    ``sqrt(pilots)`` without touching the transient, which is common to every
    replication because they all start from the same empty pipeline. This is the
    same reasoning behind Welch's procedure; the difference is only that MSER-5
    then picks the truncation point without anyone having to look at a chart.
    """
    paths = [
        system_inventory_path(_replication(config, replication))
        for replication in range(max(1, pilots))
    ]
    return mser5_truncation(np.mean(np.asarray(paths), axis=0))


@dataclass
class ScenarioOutcome:
    """Per-replication metrics for one configuration, plus the config itself."""

    name: str
    config: Dict[str, Any]
    warmup: int
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    results: List[SimulationResult] = field(default_factory=list, repr=False)
    #: Number of replications actually run. Not ``len(results)`` -- by default
    #: only the first replication's full series is retained for plotting.
    replications: int = 0

    def values(self, metric: str) -> np.ndarray:
        if metric not in self.metrics:
            raise KeyError(f"{metric!r} not recorded; have {sorted(self.metrics)}")
        return np.asarray(self.metrics[metric], dtype=float)

    def ci(self, metric: str, confidence: float = 0.95) -> ConfidenceInterval:
        return mean_ci(self.values(metric), confidence)

    def ratio_ci(self, numerator: str, denominator: str,
                 confidence: float = 0.95) -> ConfidenceInterval:
        return variance_ratio_ci(self.values(numerator), self.values(denominator), confidence)

    def summary(self, metrics: Optional[Sequence[str]] = None) -> Dict[str, ConfidenceInterval]:
        keys = list(metrics) if metrics else sorted(self.metrics)
        return {key: self.ci(key) for key in keys}


def _extract_metrics(result: SimulationResult) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    demand_var = float(np.var(result.customer_demand, ddof=1))
    metrics["var_demand"] = demand_var
    metrics["mean_demand"] = float(result.customer_demand.mean())
    for series in result.stocking_series():
        order_var = float(np.var(series.orders_placed, ddof=1))
        metrics[f"var_orders:{series.name}"] = order_var
        metrics[f"bullwhip:{series.name}"] = order_var / demand_var if demand_var > 0 else math.nan
        metrics[f"local_bullwhip:{series.name}"] = series.bullwhip()
        metrics[f"fill_rate:{series.name}"] = series.fill_rate()
        metrics[f"on_hand:{series.name}"] = float(series.on_hand.mean())
        metrics[f"backlog:{series.name}"] = float(series.backlog.mean())
    metrics["chain_bullwhip"] = result.chain_bullwhip()
    metrics["avg_cost"] = result.average_cost()
    metrics["avg_inventory"] = result.average_inventory()
    metrics["service_fill_rate"] = float(
        np.mean([s.fill_rate() for s in result.stocking_series() if s.level == 0])
    )
    return metrics


def run_scenario(
    config: Dict[str, Any],
    name: str = "scenario",
    warmup: Optional[int] = None,
    keep_results: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ScenarioOutcome:
    """Run every replication of a configuration and collect per-replication metrics.

    ``warmup`` overrides the config. Pass the *same* value to every scenario in a
    comparison -- that is what ``estimate_warmup`` on a reference config is for.
    """
    config = merge_config(DEFAULT_CONFIG, config)
    replications = int(config["run"]["replications"])
    if warmup is None:
        requested = config["run"].get("warmup", 0)
        warmup = estimate_warmup(config) if requested == "auto" else int(requested)

    outcome = ScenarioOutcome(
        name=name, config=config, warmup=int(warmup), replications=replications
    )
    for replication in range(replications):
        result = _replication(config, replication).trim(warmup)
        for key, value in _extract_metrics(result).items():
            outcome.metrics.setdefault(key, []).append(value)
        # The first replication's full series is kept for figures; holding all
        # of them is memory spent for no analytical gain.
        if keep_results or replication == 0:
            outcome.results.append(result)
        if progress is not None:
            progress(replication + 1, replications)
    return outcome


def compare_scenarios(
    treatment: ScenarioOutcome,
    control: ScenarioOutcome,
    metric: str,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Paired-t interval on ``treatment - control`` for one metric.

    Raises if the two scenarios were not run with the same seed, because the
    pairing would then be fictitious and the interval meaningless.
    """
    if treatment.config["run"]["seed"] != control.config["run"]["seed"]:
        raise ValueError(
            "paired comparison requires common random numbers: the two scenarios "
            "were run with different seeds"
        )
    if treatment.warmup != control.warmup:
        raise ValueError(
            "paired comparison requires the same warm-up truncation in both scenarios"
        )
    return paired_difference_ci(treatment.values(metric), control.values(metric), confidence)


def percent_reduction(treatment: ScenarioOutcome, control: ScenarioOutcome,
                      metric: str, confidence: float = 0.95) -> ConfidenceInterval:
    """Paired percentage change in ``metric``, treatment relative to control."""
    a = treatment.values(metric)
    b = control.values(metric)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(b) > 1e-12, 100.0 * (a - b) / b, np.nan)
    finite = ratio[np.isfinite(ratio)]
    if finite.size < 2:
        raise ValueError("not enough finite paired observations")
    return mean_ci(finite, confidence)
