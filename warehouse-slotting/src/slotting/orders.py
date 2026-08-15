"""Synthetic order streams, and the co-occurrence structure that makes affinity real.

An order generator that samples lines independently from the popularity
distribution produces a stream in which affinity slotting is worthless - by
construction, no pair of SKUs is more likely to appear together than chance.
Plenty of published slotting comparisons do exactly that and then report that
affinity adds nothing, which is a statement about the generator.

Real baskets are not independent draws. They have a *theme*: a customer
ordering shelf-stable grocery orders shelf-stable grocery. The generator here
models that with a two-component mixture per line - draw from the order's theme
family with probability ``theme_strength``, otherwise from the global
popularity distribution - which produces measurable lift on within-family pairs
and none on cross-family pairs. :meth:`OrderStream.lift_summary` reports the
realised lift so the benchmark is honest about how much structure was planted.

Basket size is a zero-truncated negative binomial. That matters more than it
sounds: single-line orders cannot benefit from affinity slotting at all, and
the share of them is the single biggest driver of how much affinity is worth.
Roodbergen & de Koster (2001) and Petersen & Aase (2004) both find the
storage-policy comparison flips with basket size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .skus import Catalog

__all__ = ["Order", "OrderStream", "OrderConfig", "generate_orders"]


@dataclass(frozen=True, slots=True)
class Order:
    """One customer order: a set of SKUs with case quantities."""

    order_id: int
    lines: tuple[int, ...]
    quantities: tuple[int, ...]

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def total_units(self) -> int:
        return int(sum(self.quantities))


@dataclass(frozen=True)
class OrderConfig:
    n_orders: int = 4_000
    mean_lines: float = 3.0
    lines_dispersion: float = 2.6  # >1 = overdispersed relative to Poisson
    max_lines: int = 28
    theme_probability: float = 0.75  # share of orders with a family theme
    theme_strength: float = 0.62  # within a themed order, share of themed lines


class OrderStream:
    """A sequence of orders plus the statistics the slotting algorithms consume."""

    def __init__(self, orders: Sequence[Order], n_skus: int) -> None:
        self.orders: tuple[Order, ...] = tuple(orders)
        self.n_skus = n_skus

    def __len__(self) -> int:
        return len(self.orders)

    def __iter__(self) -> Iterator[Order]:
        return iter(self.orders)

    def __getitem__(self, i: int) -> Order:
        return self.orders[i]

    # ------------------------------------------------------------------
    def line_counts(self) -> np.ndarray:
        """Observed pick lines per SKU."""
        counts = np.zeros(self.n_skus, dtype=float)
        for order in self.orders:
            for sku in order.lines:
                counts[sku] += 1.0
        return counts

    def unit_counts(self) -> np.ndarray:
        counts = np.zeros(self.n_skus, dtype=float)
        for order in self.orders:
            for sku, q in zip(order.lines, order.quantities):
                counts[sku] += q
        return counts

    def basket_size_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for order in self.orders:
            hist[order.n_lines] = hist.get(order.n_lines, 0) + 1
        return dict(sorted(hist.items()))

    def mean_lines(self) -> float:
        return float(np.mean([o.n_lines for o in self.orders])) if self.orders else 0.0

    def single_line_share(self) -> float:
        if not self.orders:
            return 0.0
        return float(np.mean([o.n_lines == 1 for o in self.orders]))

    # ------------------------------------------------------------------
    def cooccurrence(self, min_count: int = 1) -> dict[tuple[int, int], int]:
        """Symmetric co-occurrence counts over SKU pairs, keyed ``(i, j)`` with ``i < j``.

        Cost is sum over orders of ``C(n_lines, 2)``, which the max_lines cap
        keeps bounded. A 4,000-order stream with mean 3.4 lines gives on the
        order of 30,000 pair observations.
        """
        counts: dict[tuple[int, int], int] = {}
        for order in self.orders:
            lines = sorted(set(order.lines))
            for a in range(len(lines)):
                for b in range(a + 1, len(lines)):
                    key = (lines[a], lines[b])
                    counts[key] = counts.get(key, 0) + 1
        if min_count > 1:
            counts = {k: v for k, v in counts.items() if v >= min_count}
        return counts

    def lift_summary(self, catalog: Catalog) -> dict[str, float]:
        """Realised lift of within-family pairs against the independence baseline.

        The baseline has to account for basket-size dispersion. Comparing
        against ``P(i) * P(j) * n_orders`` reports lift above 1 even for a
        generator with no structure at all, because large orders contribute
        quadratically many pairs and are over-represented in the observed
        counts. The expectation used here is

            E[pairs(i, j)] = q_i * q_j * sum_o n_o * (n_o - 1)

        with ``q_i`` the share of all pick lines belonging to SKU i, which is
        the correct null for lines drawn independently within an order of a
        given size. A value near 1.0 means the generator planted no structure
        and affinity slotting cannot help.

        The expectation is summed over *all* pairs, not just the observed ones.
        Restricting both sides to observed pairs is a selection bias that
        reports lift near 2 on a stream with no structure whatsoever, which is
        how planted-affinity results get overstated.
        """
        counts = self.cooccurrence()
        if not counts:
            return {"within_family_lift": 1.0, "cross_family_lift": 1.0, "n_pairs": 0.0}
        line_counts = self.line_counts()
        total_lines = max(line_counts.sum(), 1.0)
        q = line_counts / total_lines
        pair_mass = float(sum(o.n_lines * (o.n_lines - 1) for o in self.orders))

        within_exp = 0.0
        for fam in np.unique(catalog.family):
            qf = q[catalog.family == fam]
            within_exp += (qf.sum() ** 2 - float((qf**2).sum())) / 2.0
        all_exp = (1.0 - float((q**2).sum())) / 2.0
        within_exp *= pair_mass
        cross_exp = (all_exp * pair_mass) - within_exp

        within_obs = cross_obs = 0.0
        for (i, j), c in counts.items():
            if catalog.family[i] == catalog.family[j]:
                within_obs += c
            else:
                cross_obs += c
        return {
            "within_family_lift": within_obs / within_exp if within_exp else 1.0,
            "cross_family_lift": cross_obs / cross_exp if cross_exp else 1.0,
            "n_pairs": float(len(counts)),
        }

    def split(self, fraction: float) -> tuple["OrderStream", "OrderStream"]:
        """Chronological split, used to fit slotting on one period and score on the next."""
        k = int(round(fraction * len(self.orders)))
        return (
            OrderStream(self.orders[:k], self.n_skus),
            OrderStream(self.orders[k:], self.n_skus),
        )


def generate_orders(
    catalog: Catalog,
    config: OrderConfig | None = None,
    seed: int = 23,
) -> OrderStream:
    """Draw a reproducible order stream against a catalogue."""
    cfg = config or OrderConfig()
    rng = np.random.default_rng(seed)
    n_skus = catalog.n_skus

    popularity = catalog.picks / catalog.picks.sum()
    global_cdf = np.cumsum(popularity)
    global_cdf[-1] = 1.0
    families = catalog.family
    family_ids = np.unique(families)
    members = {int(f): np.flatnonzero(families == f) for f in family_ids}
    family_cdf: dict[int, np.ndarray | None] = {}
    for f, idx in members.items():
        total = popularity[idx].sum()
        if total <= 0 or len(idx) == 0:
            family_cdf[f] = None
        else:
            c = np.cumsum(popularity[idx] / total)
            c[-1] = 1.0
            family_cdf[f] = c
    family_weight = np.asarray([popularity[members[int(f)]].sum() for f in family_ids])
    family_weight = family_weight / family_weight.sum()
    family_choice_cdf = np.cumsum(family_weight)
    family_choice_cdf[-1] = 1.0

    # Zero-truncated negative binomial basket size.
    mean = cfg.mean_lines
    var = mean * cfg.lines_dispersion
    if var <= mean:
        var = mean * 1.0001
    p_nb = mean / var
    r_nb = mean * p_nb / (1.0 - p_nb)

    orders: list[Order] = []
    for oid in range(cfg.n_orders):
        # Zero truncation by rejection, not by adding one: adding one shifts the
        # whole distribution and quietly destroys the single-line share, which
        # is the statistic the affinity result is most sensitive to.
        n_lines = 0
        while n_lines < 1:
            n_lines = int(rng.negative_binomial(r_nb, p_nb))
        n_lines = min(n_lines, cfg.max_lines, n_skus)

        themed = rng.random() < cfg.theme_probability
        theme = (
            int(family_ids[int(np.searchsorted(family_choice_cdf, rng.random()))])
            if themed
            else -1
        )

        chosen: list[int] = []
        seen: set[int] = set()
        attempts = 0
        while len(chosen) < n_lines and attempts < 12 * n_lines:
            attempts += 1
            if themed and rng.random() < cfg.theme_strength and family_cdf[theme] is not None:
                pool = members[theme]
                sku = int(pool[int(np.searchsorted(family_cdf[theme], rng.random()))])
            else:
                sku = int(np.searchsorted(global_cdf, rng.random()))
            if sku in seen:
                continue
            seen.add(sku)
            chosen.append(sku)

        qty = rng.integers(1, 4, size=len(chosen))
        orders.append(
            Order(order_id=oid, lines=tuple(chosen), quantities=tuple(int(q) for q in qty))
        )

    return OrderStream(orders, n_skus)
