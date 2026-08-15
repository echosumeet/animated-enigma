"""Synthetic demand generator for a product x region x channel hierarchy.

Every number in this repository comes from data produced here. The generator is
not a toy sine wave plus noise -- it reproduces the specific structures that
make real demand planning hard, and it does so with an explicit seed so any
result is reproducible from a clean clone:

* **Multiplicative seasonality** with a product-family shape and a
  region-specific phase shift, so aggregate seasonality is not simply the
  bottom-level seasonality scaled up.
* **Trend** per item, including declining items, because a benchmark where
  everything grows makes damped trend look worse than it is.
* **Promotions** on a shared calendar with an item-specific lift and a
  post-promo dip, so a naive model sees an autocorrelated shock it cannot
  explain and a feature model has something real to learn.
* **Intermittency** via a Bernoulli occurrence process with archetype-specific
  demand probability and size dispersion, which is what puts series into all
  four Syntetos-Boylan-Croston quadrants.
* **New product introductions** with a ramp, so part of the panel has short
  history and any method that assumes a full seasonal cycle must degrade
  gracefully instead of raising.
* **Discontinuations** with a decay to zero, which is the case that exposes
  Croston's inability to forget.

The data-generating process is documented in the README so the reported
accuracies are interpretable rather than decorative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hierarchy import Hierarchy, build_hierarchy

__all__ = ["DGPConfig", "DemandPanel", "generate_panel", "ARCHETYPES"]


# archetype -> (demand_probability, size_cv2, promo_intensity, base_scale)
ARCHETYPES: dict[str, tuple[float, float, float, float]] = {
    "smooth": (1.00, 0.04, 1.0, 220.0),
    "erratic": (0.95, 0.85, 1.4, 90.0),
    "intermittent": (0.34, 0.06, 0.0, 14.0),
    "lumpy": (0.24, 1.10, 0.3, 11.0),
}


@dataclass
class DGPConfig:
    """Configuration of the data-generating process."""

    n_products: int = 8
    n_regions: int = 4
    n_channels: int = 3
    n_periods: int = 260  # 5 years of weekly buckets
    season_length: int = 52
    seed: int = 20260815

    trend_scale: float = 0.35  # annualised drift spread across items
    seasonal_amplitude: float = 0.28
    promo_rate: float = 0.09  # share of weeks flagged as promotional
    promo_lift: float = 1.9
    post_promo_dip: float = 0.80
    npi_share: float = 0.12  # fraction of items introduced mid-history
    discontinued_share: float = 0.10
    noise_floor: float = 0.02
    integer_demand: bool = True
    archetype_weights: dict[str, float] = field(
        default_factory=lambda: {
            "smooth": 0.34,
            "erratic": 0.22,
            "intermittent": 0.24,
            "lumpy": 0.20,
        }
    )

    @property
    def n_bottom(self) -> int:
        return self.n_products * self.n_regions * self.n_channels


@dataclass
class DemandPanel:
    """A generated panel plus everything needed to evaluate on it."""

    y: np.ndarray  # (n_bottom, T) bottom-level demand
    keys: list[tuple[str, str, str]]
    hierarchy: Hierarchy
    promo: np.ndarray  # (n_bottom, T) 0/1 promotion flag
    archetypes: list[str]
    config: DGPConfig
    launch: np.ndarray  # first active period per series
    discontinued: np.ndarray  # first dead period per series (T if alive)

    @property
    def n_bottom(self) -> int:
        return self.y.shape[0]

    @property
    def n_periods(self) -> int:
        return self.y.shape[1]

    def node_series(self) -> np.ndarray:
        """All hierarchy nodes as a ``(n_nodes, T)`` matrix, coherent by build."""
        return self.hierarchy.aggregate(self.y)

    def node_promo(self) -> np.ndarray:
        """Promotion intensity per node: volume-weighted share on promotion."""
        S = self.hierarchy.S
        promo_units = S @ (self.promo * self.y)
        units = S @ self.y
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(units > 0, promo_units / np.maximum(units, 1e-9), 0.0)
        return np.clip(out, 0.0, 1.0)

    def describe(self) -> str:
        total = float(self.y.sum())
        zero_share = float(np.mean(self.y == 0))
        return (
            f"DemandPanel: {self.n_bottom} bottom series x {self.n_periods} periods, "
            f"{self.hierarchy.n_nodes} hierarchy nodes, total units {total:,.0f}, "
            f"zero-cell share {zero_share:.1%}"
        )


def _seasonal_index(
    m: int, rng: np.random.Generator, amplitude: float, n_harmonics: int = 3
) -> np.ndarray:
    """A positive, mean-one multiplicative seasonal profile of length m."""
    t = np.arange(m, dtype=float)
    shape = np.zeros(m)
    for k in range(1, n_harmonics + 1):
        a = rng.normal(0.0, 1.0 / k)
        b = rng.normal(0.0, 1.0 / k)
        shape += a * np.sin(2 * np.pi * k * t / m) + b * np.cos(2 * np.pi * k * t / m)
    sd = float(np.std(shape))
    if sd < 1e-9:
        return np.ones(m)
    shape = shape / sd
    idx = 1.0 + amplitude * shape
    idx = np.clip(idx, 0.15, None)
    return idx / float(np.mean(idx))


def _promo_calendar(
    T: int, m: int, rng: np.random.Generator, rate: float
) -> np.ndarray:
    """A promo calendar with seasonal clustering (peaks get more events)."""
    week = np.arange(T) % m
    seasonal_pref = 0.6 + 0.8 * (0.5 + 0.5 * np.cos(2 * np.pi * (week - 46) / m))
    p = np.clip(rate * seasonal_pref, 0.0, 0.6)
    flags = (rng.random(T) < p).astype(float)
    # promotions do not run back to back
    for t in range(1, T):
        if flags[t] > 0 and flags[t - 1] > 0:
            flags[t] = 0.0
    return flags


def generate_panel(config: DGPConfig | None = None) -> DemandPanel:
    """Generate a reproducible hierarchical demand panel."""
    cfg = config or DGPConfig()
    rng = np.random.default_rng(cfg.seed)
    T, m = cfg.n_periods, cfg.season_length

    products = [f"P{i + 1:02d}" for i in range(cfg.n_products)]
    regions = [f"R{i + 1}" for i in range(cfg.n_regions)]
    channels = ["RETAIL", "ECOM", "WHSL"][: cfg.n_channels]
    if cfg.n_channels > 3:
        channels = channels + [f"CH{i}" for i in range(3, cfg.n_channels)]

    keys = [(p, r, c) for p in products for r in regions for c in channels]
    hier = build_hierarchy(keys, dimensions=("product", "region", "channel"))

    # family-level structure shared across the items that roll up together
    product_season = {p: _seasonal_index(m, rng, cfg.seasonal_amplitude) for p in products}
    region_phase = {r: int(rng.integers(0, 5)) for r in regions}
    region_scale = {r: float(rng.uniform(0.55, 1.45)) for r in regions}
    channel_scale = dict(zip(channels, [1.0, 0.62, 1.25][: len(channels)]))
    channel_promo_sens = dict(zip(channels, [1.0, 1.35, 0.55][: len(channels)]))
    product_promo = {p: _promo_calendar(T, m, rng, cfg.promo_rate) for p in products}

    names = list(cfg.archetype_weights)
    weights = np.array([cfg.archetype_weights[n] for n in names], dtype=float)
    weights = weights / weights.sum()

    n = len(keys)
    y = np.zeros((n, T))
    promo = np.zeros((n, T))
    archetypes: list[str] = []
    launch = np.zeros(n, dtype=int)
    discontinued = np.full(n, T, dtype=int)

    for i, (p, r, c) in enumerate(keys):
        arch = str(rng.choice(names, p=weights))
        archetypes.append(arch)
        prob, cv2, promo_int, scale = ARCHETYPES[arch]

        base = (
            scale
            * region_scale[r]
            * channel_scale[c]
            * float(rng.lognormal(0.0, 0.35))
        )

        # trend: annualised growth rate, centred slightly negative for realism
        growth = float(rng.normal(-0.01, cfg.trend_scale)) / m
        trend = np.exp(growth * np.arange(T))

        season = np.roll(product_season[p], region_phase[r])
        season_path = np.tile(season, int(np.ceil(T / m)))[:T]

        flags = product_promo[p].copy()
        sens = promo_int * channel_promo_sens[c]
        if sens <= 0:
            flags[:] = 0.0
        promo[i] = flags
        lift = np.ones(T)
        item_lift = 1.0 + (cfg.promo_lift - 1.0) * sens * float(rng.uniform(0.6, 1.4))
        lift[flags > 0] = item_lift
        dip_idx = np.flatnonzero(flags > 0) + 1
        dip_idx = dip_idx[dip_idx < T]
        lift[dip_idx] *= cfg.post_promo_dip

        mu = base * trend * season_path * lift

        # lifecycle: introductions ramp in, discontinuations decay out
        life = np.ones(T)
        u = rng.random()
        if u < cfg.npi_share:
            start = int(rng.integers(m, T - 3 * m // 2))
            launch[i] = start
            ramp = np.clip((np.arange(T) - start) / (m / 3.0), 0.0, 1.0)
            life *= ramp
        elif u < cfg.npi_share + cfg.discontinued_share:
            stop = int(rng.integers(T - 2 * m, T - m // 2))
            discontinued[i] = stop
            decay = np.where(
                np.arange(T) < stop, 1.0, np.exp(-(np.arange(T) - stop) / (m / 6.0))
            )
            decay[decay < 0.02] = 0.0
            life *= decay

        mu = mu * life
        mu = np.clip(mu, 0.0, None)

        # occurrence process: promo weeks are much more likely to sell
        occ_p = np.clip(prob * (1.0 + 0.8 * (flags > 0)), 0.0, 1.0)
        occurs = rng.random(T) < occ_p
        occurs = occurs & (mu > cfg.noise_floor * base)

        # size process: gamma with CV^2 = 1/shape
        shape_k = max(0.35, 1.0 / max(cv2, 1e-3))
        sizes = np.zeros(T)
        active = np.flatnonzero(occurs)
        if active.size:
            mean_size = mu[active] / np.clip(occ_p[active], 1e-6, None)
            sizes[active] = rng.gamma(shape_k, mean_size / shape_k)

        if cfg.integer_demand:
            sizes = np.floor(sizes + rng.random(T) * (sizes > 0))
            sizes = np.clip(sizes, 0.0, None)
        y[i] = sizes

    return DemandPanel(
        y=y,
        keys=keys,
        hierarchy=hier,
        promo=promo,
        archetypes=archetypes,
        config=cfg,
        launch=launch,
        discontinued=discontinued,
    )
