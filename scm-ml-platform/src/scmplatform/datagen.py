"""Synthetic panel generator.

Everything in this repository is exercised against data produced here; there is no
external dataset. The process is a multiplicative daily demand model per SKU:

    units_t = base * seasonal(t) * trend(t) * promo_lift_t * price_elasticity_t * eps_t

with a Poisson-ish integer draw, plus two columns that are *not* knowable at decision
time -- ``returns_units`` and ``settled_margin`` -- which exist to make point-in-time
violations reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Days after the event timestamp before a column's value is actually knowable.
#: This is the declarative half of the leakage check in :mod:`scmplatform.features`.
KNOWLEDGE_DELAY_DAYS: dict[str, int] = {
    "units": 1,
    "price": 0,
    "promo_flag": 0,
    "on_hand": 1,
    "returns_units": 7,
    "settled_margin": 14,
}


@dataclass
class PanelConfig:
    n_skus: int = 40
    n_days: int = 540
    start: str = "2024-01-01"
    seed: int = 7
    regions: tuple[str, ...] = ("north", "south", "east", "west")
    drift_start_frac: float = 0.75
    drift_price_shift: float = 0.30
    categories: tuple[str, ...] = field(default=("core", "seasonal", "longtail"))


def make_panel(cfg: PanelConfig | None = None) -> pd.DataFrame:
    """Return a tidy daily panel with one row per (sku, date)."""
    cfg = cfg or PanelConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start, periods=cfg.n_days, freq="D")
    t = np.arange(cfg.n_days)
    drift_from = int(cfg.n_days * cfg.drift_start_frac)

    frames = []
    for i in range(cfg.n_skus):
        sku = f"SKU-{i:03d}"
        category = cfg.categories[i % len(cfg.categories)]
        region = cfg.regions[i % len(cfg.regions)]
        # Long-tail items are low-volume and noisier, so they are genuinely harder to
        # forecast. That asymmetry is what the slice-level monitor is meant to surface.
        base = float(rng.uniform(8, 90)) * (0.12 if category == "longtail" else 1.0)
        weekly = 1.0 + 0.25 * np.sin(2 * np.pi * (t % 7) / 7 + rng.uniform(0, 6.28))
        annual = 1.0 + (0.35 if category == "seasonal" else 0.08) * np.sin(
            2 * np.pi * t / 365.0 + rng.uniform(0, 6.28)
        )
        trend = 1.0 + rng.uniform(-0.15, 0.30) * (t / cfg.n_days)

        promo = (rng.random(cfg.n_days) < 0.07).astype(int)
        price0 = float(rng.uniform(4.0, 45.0))
        price = price0 * (1 - 0.20 * promo) * rng.normal(1.0, 0.02, cfg.n_days)
        # Post-drift regime: a permanent price/cost step change on part of the range.
        if i % 3 == 0:
            price[drift_from:] *= 1 + cfg.drift_price_shift

        elasticity = (price / price0) ** -1.3
        lam = np.clip(base * weekly * annual * trend * elasticity * (1 + 0.9 * promo), 0.5, None)
        units = rng.poisson(lam).astype(float)

        on_hand = np.maximum(0.0, np.cumsum(rng.normal(0, 6, cfg.n_days)) + base * 6)
        returns = rng.binomial(units.astype(int), 0.04).astype(float)
        settled_margin = units * price * rng.uniform(0.18, 0.34) - returns * price

        frames.append(
            pd.DataFrame(
                {
                    "sku": sku,
                    "date": dates,
                    "region": region,
                    "category": category,
                    "units": units,
                    "price": price,
                    "promo_flag": promo,
                    "on_hand": on_hand,
                    "returns_units": returns,
                    "settled_margin": settled_margin,
                }
            )
        )

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["sku", "date"], ignore_index=True)


def inject_quality_faults(panel: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Return a copy of ``panel`` with realistic upstream data faults inserted.

    The faults mirror what actually shows up in planning feeds: nulls where a join
    dropped, negative quantities from an un-netted return, a duplicated primary key
    from a replayed extract, and a price outlier from a units-of-measure mistake.
    """
    rng = np.random.default_rng(seed)
    out = panel.copy()
    n = len(out)
    out.loc[rng.choice(n, size=max(1, n // 400), replace=False), "price"] = np.nan
    out.loc[rng.choice(n, size=max(1, n // 600), replace=False), "units"] = -3.0
    out.loc[rng.choice(n, size=max(1, n // 900), replace=False), "price"] = 9_999.0
    dupes = out.iloc[rng.choice(n, size=5, replace=False)]
    return pd.concat([out, dupes], ignore_index=True)
