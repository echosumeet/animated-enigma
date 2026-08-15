"""Bullwhip measurement, its analytic bounds, and a decomposition that adds up.

Measurement
-----------
Amplification is ``Var(orders placed) / Var(demand received)``. Two versions are
reported and they answer different questions:

* **local** -- against the node's own incoming order stream. "Am I making it
  worse?"
* **cumulative** -- against true end-customer demand. "How distorted is the
  signal by the time it reaches me?" This is the one that sizes the factory.

Validation
----------
For a single stage with i.i.d. demand and an order-up-to policy whose target is
``S_t = F_t * L`` (constant safety term), the order in period ``t`` is exactly

    ``q_t = d_t + L * (F_t - F_{t-1})``

because the inventory position just before the review is ``S_{t-1} - d_t``.
Substituting each forecaster's recursion gives a closed form:

* moving average of ``p`` periods, ``F_t - F_{t-1} = (d_t - d_{t-p})/p``, so
  ``Var(q)/Var(d) = 1 + 2L/p + 2L^2/p^2`` -- exactly Theorem 1 of Chen, Drezner,
  Ryan & Simchi-Levi (2000);
* exponential smoothing, ``F_t - F_{t-1} = alpha*(d_t - F_{t-1})`` with
  ``Var(F) = alpha*sigma^2/(2-alpha)``, so
  ``Var(q)/Var(d) = (1 + alpha*L)^2 + alpha^3*L^2/(2-alpha)``
  ``            = 1 + 2*alpha*L + 2*alpha^2*L^2/(2-alpha)``.

The second expression is derived here for this package's timing convention (see
``Node.protection_interval``); it is the exponential-smoothing analogue of the
moving-average theorem and reduces to it under the usual ``alpha ~ 1/p``
correspondence. It is quoted as a derivation, not as a citation.

The test suite runs the *full* simulator -- engine, network, event loop and all
-- against both. An independent analytic special case is the strongest
validation a simulator of this kind can have, and it catches the entire class of
off-by-one timing bugs that otherwise go unnoticed for years. It is also why
``allow_returns`` exists: the closed form permits negative orders, and clamping
them at zero biases the measured variance downward by several percent at high
``alpha``.

Decomposition
-------------
The three mechanisms of interest -- demand signal processing, order batching and
lead time -- are not additive and do not commute, so "turn one off and subtract"
gives an answer that depends on the order you turn them off in. Instead all
``2^3`` combinations are simulated and each mechanism is credited with its
**Shapley value** on the log of the amplification ratio. Log scale because the
mechanisms compose multiplicatively; Shapley because it is the unique
attribution that is efficient (the parts sum to the whole), symmetric, and
independent of ablation order.

References
----------
Lee, Padmanabhan & Whang (1997), "Information distortion in a supply chain: the
bullwhip effect", Management Science 43(4), 546-558.
Chen, Drezner, Ryan & Simchi-Levi (2000), "Quantifying the bullwhip effect in a
simple supply chain", Management Science 46(3), 436-443.
Chen, Ryan & Simchi-Levi (2000), "The impact of exponential smoothing forecasts
on the bullwhip effect", Naval Research Logistics 47(4), 269-286.
Shapley (1953), "A value for n-person games".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .experiments import DEFAULT_CONFIG, ScenarioOutcome, merge_config, run_scenario
from .metrics import ConfidenceInterval, mean_ci

__all__ = [
    "chen_moving_average_bullwhip",
    "exponential_smoothing_bullwhip",
    "EchelonAmplification",
    "measure_by_echelon",
    "MECHANISMS",
    "DecompositionResult",
    "decompose_bullwhip",
    "smoothing_sweep",
]


def chen_moving_average_bullwhip(protection: float, window: int) -> float:
    """``1 + 2L/p + 2L^2/p^2`` -- Chen, Drezner, Ryan & Simchi-Levi (2000), Thm 1."""
    if window < 1:
        raise ValueError("moving-average window must be >= 1")
    ratio = protection / window
    return 1.0 + 2.0 * ratio + 2.0 * ratio * ratio


def exponential_smoothing_bullwhip(protection: float, alpha: float) -> float:
    """``1 + 2aL + 2a^2 L^2/(2-a)`` -- derived in the module docstring.

    Note the ``L^2`` term: amplification grows *quadratically* in the protection
    interval. Halving a lead time cuts the variance the next echelon up has to
    absorb by roughly four, which is the quantitative reason lead-time reduction
    is worth far more than it looks on a safety-stock spreadsheet -- there the
    benefit is only ``sqrt(L)``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return (
        1.0
        + 2.0 * alpha * protection
        + (2.0 * alpha ** 2 * protection ** 2) / (2.0 - alpha)
    )


@dataclass
class EchelonAmplification:
    """Amplification by echelon, with intervals, for one configuration."""

    scenario: ScenarioOutcome
    node_order: List[str]
    local: Dict[str, ConfidenceInterval]
    cumulative: Dict[str, ConfidenceInterval]

    def table(self) -> List[Tuple[str, float, float, float, float]]:
        rows = []
        for name in self.node_order:
            local = self.local[name]
            cumulative = self.cumulative[name]
            rows.append((name, local.mean, local.half_width, cumulative.mean, cumulative.half_width))
        return rows


def measure_by_echelon(
    config: Dict[str, Any],
    name: str = "bullwhip",
    warmup: Optional[int] = None,
) -> EchelonAmplification:
    """Run a configuration and report amplification at every stocking echelon."""
    outcome = run_scenario(config, name=name, warmup=warmup, keep_results=True)
    reference = outcome.results[0]
    order = [s.name for s in reference.stocking_series()]
    local = {n: outcome.ci(f"local_bullwhip:{n}") for n in order}
    cumulative = {n: outcome.ratio_ci(f"var_orders:{n}", "var_demand") for n in order}
    return EchelonAmplification(outcome, order, local, cumulative)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

#: The three mechanisms, each as (on-config, off-config) overrides.
MECHANISMS: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {
    "signal": (
        {"forecast": {"kind": "exponential", "alpha": 0.3}},
        {"forecast": {"kind": "oracle"}},
    ),
    "batching": (
        {"review_period": 4, "policy": {"batch_multiple": 100.0}},
        {"review_period": 1, "policy": {"batch_multiple": 1.0}},
    ),
    "leadtime": (
        {"leadtime": {"mean": 4.0}},
        {"leadtime": {"mean": 1.0}},
    ),
}


@dataclass
class DecompositionResult:
    """Shapley attribution of log-amplification to each mechanism."""

    baseline: ConfidenceInterval
    full: ConfidenceInterval
    contributions: Dict[str, ConfidenceInterval]
    multipliers: Dict[str, float]
    cell_means: Dict[Tuple[str, ...], float]
    metric: str

    def check_additivity(self, tolerance: float = 1e-6) -> None:
        """The Shapley values must reconstruct the full effect exactly."""
        total = math.log(self.full.mean) - math.log(self.baseline.mean)
        parts = sum(ci.mean for ci in self.contributions.values())
        if abs(total - parts) > tolerance:  # pragma: no cover - guard
            raise AssertionError(
                f"Shapley decomposition does not add up: {parts:.9f} vs {total:.9f}"
            )

    def table(self) -> List[Tuple[str, float, float, float, float]]:
        """``(mechanism, log contribution, half width, multiplier, share %)``."""
        total = sum(ci.mean for ci in self.contributions.values())
        rows = []
        for key, interval in self.contributions.items():
            share = 100.0 * interval.mean / total if abs(total) > 1e-12 else float("nan")
            rows.append((key, interval.mean, interval.half_width,
                         self.multipliers[key], share))
        return rows


def _geometric_ci(log_values: np.ndarray) -> ConfidenceInterval:
    """Geometric-mean interval from log-scale replications.

    Reported as a geometric mean rather than an arithmetic one so that
    ``log(full) - log(baseline)`` is exactly the sum of the Shapley
    contributions. An arithmetic mean of ratios would break that identity by a
    Jensen gap and quietly make the decomposition stop adding up.
    """
    interval = mean_ci(log_values)
    centre = float(np.exp(interval.mean))
    return ConfidenceInterval(
        mean=centre,
        half_width=centre * interval.half_width,
        low=float(np.exp(interval.low)),
        high=float(np.exp(interval.high)),
        n=interval.n,
        note="geometric mean of per-replication ratios",
    )


def _shapley_weights(n: int) -> Dict[int, float]:
    return {
        size: math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
        for size in range(n)
    }


def decompose_bullwhip(
    base_config: Optional[Dict[str, Any]] = None,
    mechanisms: Optional[Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
    metric: str = "chain_bullwhip",
    warmup: Optional[int] = None,
) -> DecompositionResult:
    """Simulate every on/off combination and attribute log-amplification by Shapley value.

    All ``2^k`` cells share the seed, so every cell sees the same customer demand
    in replication ``i``. The Shapley value is computed **per replication** and
    the confidence interval is taken across replications, which is only
    legitimate because of that pairing.
    """
    base = merge_config(DEFAULT_CONFIG, base_config or {})
    mechanisms = mechanisms or MECHANISMS
    keys = list(mechanisms)
    n = len(keys)
    if n == 0:
        raise ValueError("need at least one mechanism")

    if warmup is None:
        on_all = base
        for key in keys:
            on_all = merge_config(on_all, mechanisms[key][0])
        from .experiments import estimate_warmup

        warmup = estimate_warmup(on_all, pilots=3)

    cells: Dict[Tuple[str, ...], np.ndarray] = {}
    for mask in itertools.product([False, True], repeat=n):
        active = tuple(key for key, flag in zip(keys, mask) if flag)
        config = base
        for key, flag in zip(keys, mask):
            config = merge_config(config, mechanisms[key][0 if flag else 1])
        outcome = run_scenario(config, name="+".join(active) or "none", warmup=warmup)
        values = outcome.values(metric)
        cells[active] = np.log(np.maximum(values, 1e-12))

    weights = _shapley_weights(n)
    replications = len(next(iter(cells.values())))
    contributions: Dict[str, np.ndarray] = {}
    for index, key in enumerate(keys):
        total = np.zeros(replications, dtype=float)
        others = [k for k in keys if k != key]
        for size in range(n):
            for subset in itertools.combinations(others, size):
                with_key = tuple(k for k in keys if k in set(subset) | {key})
                without_key = tuple(k for k in keys if k in set(subset))
                total += weights[size] * (cells[with_key] - cells[without_key])
        contributions[key] = total

    full_key = tuple(keys)
    baseline_ci = _geometric_ci(cells[tuple()])
    full_ci = _geometric_ci(cells[full_key])
    contribution_cis = {k: mean_ci(v) for k, v in contributions.items()}
    result = DecompositionResult(
        baseline=baseline_ci,
        full=full_ci,
        contributions=contribution_cis,
        multipliers={k: float(np.exp(v.mean)) for k, v in contribution_cis.items()},
        cell_means={k: float(np.exp(v.mean())) for k, v in cells.items()},
        metric=metric,
    )
    # Efficiency holds on the log-mean scale by construction; assert it rather
    # than trust it, because a mis-indexed subset would silently redistribute
    # attribution without changing the total.
    total_log = float(np.mean(cells[full_key] - cells[tuple()]))
    parts_log = float(sum(v.mean for v in contribution_cis.values()))
    if abs(total_log - parts_log) > 1e-8:  # pragma: no cover - guard
        raise AssertionError("Shapley decomposition failed the efficiency check")
    return result


def smoothing_sweep(
    alphas: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.8),
    base_config: Optional[Dict[str, Any]] = None,
    warmup: Optional[int] = None,
) -> List[Tuple[float, ConfidenceInterval, float]]:
    """Chain amplification as a function of the smoothing constant.

    Returns ``(alpha, simulated chain bullwhip CI, single-stage analytic value)``.
    The analytic column is the *single-stage* Chen bound at the retailer's
    protection interval -- it is there as a sanity anchor, not as a prediction of
    the multi-echelon number, which compounds.
    """
    base = merge_config(DEFAULT_CONFIG, base_config or {})
    rows = []
    for alpha in alphas:
        config = merge_config(base, {"forecast": {"kind": "exponential", "alpha": alpha}})
        outcome = run_scenario(config, name=f"alpha={alpha}", warmup=warmup)
        protection = float(config["leadtime"]["mean"]) + float(
            config["leadtime"]["order_lead_time"]
        ) + float(config["review_period"]) - 1.0
        rows.append(
            (
                float(alpha),
                outcome.ratio_ci("var_orders:retailer", "var_demand"),
                exponential_smoothing_bullwhip(protection, alpha),
            )
        )
    return rows
