"""Clark-Scarf decomposition for a serial multi-echelon system.

Clark and Scarf's 1960 result is that an ``N``-stage serial system under periodic
review with backorders decomposes into ``N`` single-stage problems, solved from
the customer-facing stage upstream, provided you work in *echelon* rather than
installation stock.  Each upstream stage inherits an "induced penalty" - the cost
its shortfall imposes downstream - and its optimal echelon base-stock level is
then a one-dimensional minimisation.

Why this matters operationally: it is the formal reason that pushing safety stock
upstream is cheap and pushing it downstream is expensive, and it gives the exact
tradeoff rather than a rule of thumb.  It is also the reason installation-stock
reorder points set independently per site are wrong - they double-count the same
demand risk at every level, and the resulting network holds far more inventory
than the same service needs.

Timing convention (stated explicitly, because most of the published disagreement
between formulations is really disagreement about timing):

* Each period: receive arrivals, place orders, then demand occurs.
* Holding and backorder costs are charged on **end-of-period** echelon inventory
  levels.  Stage ``i``'s risk period is therefore ``L_i + 1`` periods.
* The order stage ``i-1`` places when stage ``i``'s shipment lands is capped by
  echelon ``i``'s inventory level at that moment, which is ``y_i`` minus demand
  over ``L_i`` periods.

Cost accounting uses echelon holding costs ``e_i = h_i - h_{i+1}`` charged on
echelon inventory *levels* (which can be negative), plus a penalty of
``p + H`` on stage-1 backorders where ``H = sum_i e_i`` is the installation
holding cost at stage 1.  The ``H`` term is not a fudge: charging ``e_i`` on a
negative echelon level already credits back holding cost on backordered units, so
it has to be added to the penalty to recover the true cost.

References
----------
Clark, A.J. and Scarf, H. (1960) 'Optimal policies for a multi-echelon inventory
problem', *Management Science* 6(4), 475-490.
Chen, F. and Zheng, Y.-S. (1994) 'Lower bounds for multi-echelon stochastic
inventory systems', *Management Science* 40(11), 1426-1443.
Zipkin, P.H. (2000) *Foundations of Inventory Management*, Ch. 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

__all__ = [
    "SerialStage",
    "ClarkScarfSolution",
    "clark_scarf",
    "simulate_serial_system",
    "discretised_demand_pmf",
]


@dataclass(frozen=True)
class SerialStage:
    """One stage of a serial system, indexed from the customer upstream.

    ``echelon_holding`` is the *incremental* value added at this stage per unit
    per period, ``e_i = h_i - h_{i+1}``.  ``lead_time`` is the transit time from
    the next stage upstream (or from the external supplier, for the last stage).
    """

    name: str
    lead_time: int
    echelon_holding: float

    def __post_init__(self) -> None:
        if self.lead_time < 0:
            raise ValueError("lead_time must be non-negative")
        if self.echelon_holding < 0:
            raise ValueError("echelon holding cost must be non-negative")


@dataclass
class ClarkScarfSolution:
    stages: list[SerialStage]
    echelon_base_stock: list[float]
    optimal_cost: float
    stage_costs: list[float]
    grid: np.ndarray

    def installation_holding_costs(self) -> list[float]:
        """Convert echelon holding costs back to installation holding costs."""
        out: list[float] = []
        running = 0.0
        for st in reversed(self.stages):
            running += st.echelon_holding
            out.append(running)
        return list(reversed(out))

    def summary(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for st, s, c in zip(self.stages, self.echelon_base_stock, self.stage_costs):
            rows.append(
                {
                    "stage": st.name,
                    "lead_time": float(st.lead_time),
                    "echelon_holding": float(st.echelon_holding),
                    "echelon_base_stock": float(s),
                    "stage_optimal_cost": float(c),
                }
            )
        return rows


def discretised_demand_pmf(
    mean: float, sd: float, periods: int, max_sigma: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """Integer-support pmf of demand over ``periods`` periods.

    A normal is discretised onto integers and truncated below zero, then
    renormalised.  Discretising rather than integrating keeps the dynamic program
    exact on its own support - the approximation is entirely in the demand model,
    not in the optimisation, which is the right place for it to be.
    """
    if periods <= 0:
        return np.array([0]), np.array([1.0])
    mu = mean * periods
    sigma = sd * np.sqrt(periods)
    lo = max(0, int(np.floor(mu - max_sigma * sigma)))
    hi = int(np.ceil(mu + max_sigma * sigma))
    support = np.arange(lo, hi + 1)
    pmf = stats.norm.cdf(support + 0.5, mu, sigma) - stats.norm.cdf(support - 0.5, mu, sigma)
    if lo == 0:
        pmf[0] = stats.norm.cdf(0.5, mu, sigma)
    pmf = np.maximum(pmf, 0.0)
    pmf /= pmf.sum()
    return support, pmf


def _loss_on_grid(grid: np.ndarray, support: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    """``E[(D - y)^+]`` for every ``y`` in ``grid``."""
    diff = support[None, :] - grid[:, None]
    return (np.maximum(diff, 0.0) * pmf[None, :]).sum(axis=1)


def _eval_with_extrapolation(grid: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation on the grid, linear extrapolation off the left edge.

    The cost functions are asymptotically linear to the left (once you are deep in
    backorder territory, one more unit short costs exactly ``p + H - e``), so a
    two-point extrapolation is exact there rather than merely convenient.
    """
    step = float(grid[1] - grid[0])
    left_slope = float(values[1] - values[0]) / step
    out = np.interp(x, grid, values)
    below = x < grid[0]
    if np.any(below):
        out = np.where(below, values[0] + (x - grid[0]) * left_slope, out)
    return out


def clark_scarf(
    stages: Sequence[SerialStage],
    demand_mean: float,
    demand_sd: float,
    penalty: float,
    grid_lo: float | None = None,
    grid_hi: float | None = None,
    grid_step: int = 1,
) -> ClarkScarfSolution:
    """Solve a serial system for optimal echelon base-stock levels.

    ``stages[0]`` faces the customer.  Returns echelon base-stock levels in the
    same order and the exact expected cost per period of the optimal policy under
    the discretised demand model.
    """
    if not stages:
        raise ValueError("at least one stage is required")
    if penalty <= 0:
        raise ValueError("penalty must be positive")

    H = float(sum(st.echelon_holding for st in stages))
    total_lead = sum(st.lead_time for st in stages) + 1
    mu_total = demand_mean * total_lead
    sd_total = demand_sd * np.sqrt(total_lead)
    if grid_hi is None:
        grid_hi = float(mu_total + 8.0 * sd_total)
    if grid_lo is None:
        grid_lo = float(-4.0 * demand_sd * np.sqrt(stages[0].lead_time + 1) - 2.0 * demand_mean)
    grid = np.arange(int(np.floor(grid_lo)), int(np.ceil(grid_hi)) + 1, grid_step, dtype=float)

    # --- Stage 1: the classical newsvendor with penalty (p + H) ---
    st1 = stages[0]
    sup1, pmf1 = discretised_demand_pmf(demand_mean, demand_sd, st1.lead_time + 1)
    mu_risk1 = float((sup1 * pmf1).sum())
    loss1 = _loss_on_grid(grid, sup1, pmf1)
    C = st1.echelon_holding * (grid - mu_risk1) + (penalty + H) * loss1

    base_stock: list[float] = []
    stage_costs: list[float] = []
    idx = int(np.argmin(C))
    base_stock.append(float(grid[idx]))
    stage_costs.append(float(C[idx]))
    y_star = float(grid[idx])

    # --- Upstream stages: induced-penalty recursion ---
    for st in stages[1:]:
        sup_t, pmf_t = discretised_demand_pmf(demand_mean, demand_sd, st.lead_time)
        sup_h, pmf_h = discretised_demand_pmf(demand_mean, demand_sd, st.lead_time + 1)
        mu_hold = float((sup_h * pmf_h).sum())

        arg = np.minimum(y_star, grid[:, None] - sup_t[None, :])
        downstream = _eval_with_extrapolation(grid, C, arg.ravel()).reshape(arg.shape)
        expected_downstream = (downstream * pmf_t[None, :]).sum(axis=1)

        C = st.echelon_holding * (grid - mu_hold) + expected_downstream
        idx = int(np.argmin(C))
        y_star = float(grid[idx])
        base_stock.append(y_star)
        stage_costs.append(float(C[idx]))

    return ClarkScarfSolution(
        stages=list(stages),
        echelon_base_stock=base_stock,
        optimal_cost=stage_costs[-1],
        stage_costs=stage_costs,
        grid=grid,
    )


def simulate_serial_system(
    stages: Sequence[SerialStage],
    echelon_base_stock: Sequence[float],
    demand_mean: float,
    demand_sd: float,
    penalty: float,
    n_periods: int = 40_000,
    warmup: int = 500,
    seed: int = 909,
) -> dict[str, float]:
    """Simulate the echelon base-stock policy and measure realised cost.

    This exists to check the dynamic program rather than to replace it.  The
    simulated cost should land on ``ClarkScarfSolution.optimal_cost`` within Monte
    Carlo error; if it does not, either the recursion or the timing convention is
    wrong, and both failure modes are easy to introduce and hard to spot by
    inspection.

    Demand is discretised the same way as in the DP so the two are comparing the
    same model.
    """
    n = len(stages)
    if len(echelon_base_stock) != n:
        raise ValueError("one base-stock level per stage is required")
    rng = np.random.default_rng(seed)
    sup, pmf = discretised_demand_pmf(demand_mean, demand_sd, 1)
    demand = rng.choice(sup, size=n_periods, p=pmf).astype(float)

    H = float(sum(st.echelon_holding for st in stages))
    on_hand = np.zeros(n, dtype=float)
    on_hand[-1] = float(echelon_base_stock[-1])
    backorder = 0.0
    backorder_sum = 0.0
    # pipeline[i] holds (arrival period, qty) for material moving into stage i
    pipeline: list[list[tuple[int, float]]] = [[] for _ in range(n)]

    total_cost = 0.0
    total_demand = 0.0
    filled = 0.0
    stockout_periods = 0
    counted = 0

    def echelon_level(i: int) -> float:
        """On-hand at stages i..0 plus in-transit into stages i-1..0, less backorders."""
        lvl = float(on_hand[: i + 1].sum()) - backorder
        for j in range(i):
            lvl += sum(q for _, q in pipeline[j])
        return lvl

    for t in range(n_periods):
        # 1. receipts
        for i in range(n):
            arrived = [(a, q) for a, q in pipeline[i] if a <= t]
            if arrived:
                pipeline[i] = [(a, q) for a, q in pipeline[i] if a > t]
                on_hand[i] += sum(q for _, q in arrived)
        # clear backorders at stage 0
        if backorder > 0 and on_hand[0] > 0:
            served = min(backorder, on_hand[0])
            backorder -= served
            on_hand[0] -= served

        # 2. orders, from the most upstream stage down
        for i in range(n - 1, -1, -1):
            eip = echelon_level(i) + sum(q for _, q in pipeline[i])
            want = echelon_base_stock[i] - eip
            if want <= 0:
                continue
            avail = float("inf") if i == n - 1 else float(on_hand[i + 1])
            q = min(want, avail)
            if q > 0:
                if i < n - 1:
                    on_hand[i + 1] -= q
                pipeline[i].append((t + max(1, stages[i].lead_time), q))

        # 3. demand at stage 0
        d = float(demand[t])
        served = min(on_hand[0], d)
        on_hand[0] -= served
        unmet = d - served
        backorder += unmet

        # 4. end-of-period costs on echelon inventory levels
        if t >= warmup:
            counted += 1
            total_demand += d
            filled += served
            period_cost = 0.0
            for i, st in enumerate(stages):
                period_cost += st.echelon_holding * echelon_level(i)
            period_cost += (penalty + H) * backorder
            total_cost += period_cost
            backorder_sum += backorder
            if unmet > 1e-9:
                stockout_periods += 1

    return {
        "avg_cost_per_period": float(total_cost / counted),
        "fill_rate": float(filled / total_demand) if total_demand else 1.0,
        "ready_rate": float(1.0 - stockout_periods / counted),
        "avg_backorder": float(backorder_sum / counted),
        "periods": float(counted),
    }
