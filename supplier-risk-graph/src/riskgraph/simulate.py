"""Monte Carlo disruption simulation over the multi-tier network.

Each trial samples, per site, whether a disruption starts inside the horizon, when it
starts and how long recovery takes. Overlapping outages matter, so the year is cut at
event boundaries and the availability kernel is evaluated on each interval; lost
revenue is the integral of the shortfall over time rather than a point estimate.

Recovery duration is lognormal around the site's mean recovery time. Disruption
severity is heavy tailed in practice (Sheffi, *The Resilient Enterprise*); a symmetric
distribution understates the tail that actually drives mitigation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping

import numpy as np

from .flow import output_fraction, topo_order
from .model import SupplyNetwork

HORIZON = 365.0


@dataclass
class SimResult:
    trials: int
    horizon_days: float
    expected_loss: float
    p95_loss: float
    max_loss: float
    p_any_impact: float
    mean_ttr_days: float
    p95_ttr_days: float
    mean_service_level: float

    def as_dict(self) -> dict:
        return asdict(self)


def _sample_events(net: SupplyNetwork, rng: np.random.Generator, horizon: float, sigma: float):
    """Return (site_id, start, end) for every disruption starting inside the horizon."""
    events = []
    for s in net.sites:
        if rng.random() >= s.disruption_rate * horizon / 365.0:
            continue
        start = float(rng.uniform(0.0, horizon))
        mu = np.log(s.mean_recovery_days) - 0.5 * sigma**2
        dur = float(rng.lognormal(mu, sigma))
        events.append((s.site_id, start, min(horizon, start + dur)))
    return events


def simulate(
    net: SupplyNetwork,
    trials: int = 2000,
    seed: int = 11,
    flex: float = 1.25,
    sigma: float = 0.55,
    horizon: float = HORIZON,
    buffers: Mapping[str, float] | None = None,
) -> SimResult:
    """Run the disruption simulation and return loss and recovery statistics.

    `buffers` maps part_id -> days of cover. A buffered part is unaffected until its
    cumulative impaired time in the trial exceeds the cover, which is how safety stock
    actually behaves: it buys time, it does not remove the exposure.

    `flex` is how far surviving qualified sources can be pushed above their allocation
    during a recovery. It defaults to 1.25 rather than 1.0 because zero ramp-up is not
    a neutral assumption -- it is the assumption under which dual sourcing never pays.

    Reported time-to-recover is the longest single shortfall episode in a trial, not
    the span from the first to the last event: two unrelated outages in one year are
    two recoveries, and averaging across the gap between them flatters nothing.
    """
    rng = np.random.default_rng(seed)
    fgs = net.finished_goods()
    orders = {fg.part_id: topo_order(net, fg.part_id) for fg in fgs}
    rates = {fg.part_id: fg.annual_revenue / horizon for fg in fgs}
    buffers = dict(buffers or {})
    buffered_sites = {
        pid: {e.site_id for e in net.sources(pid)} for pid in buffers if buffers[pid] > 0
    }

    losses = np.zeros(trials)
    ttrs = np.zeros(trials)
    for t in range(trials):
        events = _sample_events(net, rng, horizon, sigma)
        if not events:
            continue
        bounds = sorted({0.0, horizon} | {e[1] for e in events} | {e[2] for e in events})
        used = {pid: 0.0 for pid in buffered_sites}
        episodes: list[list[float]] = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            dt = b - a
            if dt <= 0:
                continue
            down = {sid for sid, st, en in events if st <= a < en}
            if not down:
                continue
            exempt = []
            for pid, sids in buffered_sites.items():
                if sids & down and used[pid] < buffers[pid]:
                    exempt.append(pid)
                    used[pid] += dt
            shortfall = 0.0
            for fg in fgs:
                frac = output_fraction(net, fg.part_id, down, flex, exempt, orders[fg.part_id])
                shortfall += rates[fg.part_id] * (1.0 - frac) * dt
            if shortfall > 0:
                if episodes and abs(episodes[-1][1] - a) < 1e-9:
                    episodes[-1][1] = b
                else:
                    episodes.append([a, b])
            losses[t] += shortfall
        if episodes:
            ttrs[t] = max(b - a for a, b in episodes)

    impacted = ttrs > 0
    total_revenue = sum(fg.annual_revenue for fg in fgs)
    return SimResult(
        trials=trials,
        horizon_days=horizon,
        expected_loss=float(losses.mean()),
        p95_loss=float(np.percentile(losses, 95)),
        max_loss=float(losses.max()),
        p_any_impact=float(impacted.mean()),
        mean_ttr_days=float(ttrs[impacted].mean()) if impacted.any() else 0.0,
        p95_ttr_days=float(np.percentile(ttrs[impacted], 95)) if impacted.any() else 0.0,
        mean_service_level=1.0 - float(losses.mean()) / total_revenue if total_revenue else 1.0,
    )
