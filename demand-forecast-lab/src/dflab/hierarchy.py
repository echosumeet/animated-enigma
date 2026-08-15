"""Hierarchical / grouped forecast reconciliation.

A planning organisation does not consume one forecast. Finance consumes the
total, category managers consume product-level, the DC network consumes
region-level, and replenishment consumes the bottom SKU x region x channel
cell. If those numbers are produced independently they will not add up, and the
first thing that happens in the S&OP meeting is an argument about whose number
is wrong instead of a decision. **Coherency** -- ``y_agg = S @ y_bottom`` at
every level -- is a hard requirement, not a nicety.

All four reconciliation methods here are linear maps of the base forecasts::

    y_tilde = S @ G @ y_hat

with ``S`` the summing matrix and ``G`` the method-specific mapping. Because
``S @ G @ S = S`` for every G implemented here, reconciled forecasts are
coherent by construction and the property is tested directly.

Methods
-------
bottom_up      G selects the bottom rows.
top_down       G maps the top-level forecast down by historical proportions.
ols            G = (S'S)^-1 S'  (Hyndman et al. 2011).
mint           G = (S'W^-1 S)^-1 S'W^-1 with a shrunk residual covariance
               (Wickramasuriya, Athanasopoulos & Hyndman 2019).

References
----------
Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G. & Shang, H.L. (2011). "Optimal
combination forecasts for hierarchical time series." *Computational Statistics
& Data Analysis* 55(9), 2579-2589.

Wickramasuriya, S.L., Athanasopoulos, G. & Hyndman, R.J. (2019). "Optimal
forecast reconciliation for hierarchical and grouped time series through trace
minimization." *JASA* 114(526), 804-819.

Schafer, J. & Strimmer, K. (2005). "A shrinkage approach to large-scale
covariance matrix estimation." *Statistical Applications in Genetics and
Molecular Biology* 4(1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Hierarchy",
    "build_hierarchy",
    "shrink_covariance",
    "reconcile",
    "coherency_error",
]


@dataclass
class Hierarchy:
    """A grouped time series structure and its summing matrix.

    ``S`` has shape ``(n_total, n_bottom)``; the last ``n_bottom`` rows are the
    identity block, so ``S @ b`` returns aggregates followed by the bottom
    series in their original order.
    """

    S: np.ndarray
    node_names: list[str]
    node_levels: list[str]
    bottom_keys: list[tuple[str, ...]]
    dimensions: list[str] = field(default_factory=list)

    @property
    def n_bottom(self) -> int:
        return self.S.shape[1]

    @property
    def n_nodes(self) -> int:
        return self.S.shape[0]

    def aggregate(self, bottom: np.ndarray) -> np.ndarray:
        """Map bottom-level values (n_bottom,) or (n_bottom, T) to all nodes."""
        b = np.asarray(bottom, dtype=float)
        if b.shape[0] != self.n_bottom:
            raise ValueError(
                f"expected {self.n_bottom} bottom rows, got {b.shape[0]}"
            )
        return self.S @ b

    def level_index(self, level: str) -> np.ndarray:
        """Row indices of every node at a given level name."""
        return np.array(
            [i for i, lv in enumerate(self.node_levels) if lv == level], dtype=int
        )

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for lv in self.node_levels:
            counts[lv] = counts.get(lv, 0) + 1
        parts = [f"{k}={v}" for k, v in counts.items()]
        return (
            f"Hierarchy({self.n_nodes} nodes, {self.n_bottom} bottom): "
            + ", ".join(parts)
        )


def build_hierarchy(
    keys: list[tuple[str, ...]],
    dimensions: tuple[str, ...] = ("product", "region", "channel"),
    include_pairs: bool = True,
) -> Hierarchy:
    """Build S for a grouped structure over the given key dimensions.

    ``keys`` are the bottom-level coordinate tuples, e.g. ``("P01", "R2", "ECOM")``.
    Levels generated: ``total``, each single dimension, optionally each pair of
    dimensions, then ``bottom``.

    This is a *grouped* rather than strictly hierarchical series -- product,
    region and channel are crossed, not nested -- which is the normal shape of
    a real commercial hierarchy and the case MinT was designed to handle.
    """
    if not keys:
        raise ValueError("no bottom-level keys")
    d = len(dimensions)
    if any(len(k) != d for k in keys):
        raise ValueError("every key must have one entry per dimension")

    n_bottom = len(keys)
    rows: list[np.ndarray] = []
    names: list[str] = []
    levels: list[str] = []

    # total
    rows.append(np.ones(n_bottom))
    names.append("TOTAL")
    levels.append("total")

    def add_group(dim_idx: tuple[int, ...], level_name: str) -> None:
        seen: dict[tuple[str, ...], np.ndarray] = {}
        order: list[tuple[str, ...]] = []
        for j, k in enumerate(keys):
            gk = tuple(k[i] for i in dim_idx)
            if gk not in seen:
                seen[gk] = np.zeros(n_bottom)
                order.append(gk)
            seen[gk][j] = 1.0
        for gk in sorted(order):
            rows.append(seen[gk])
            names.append("/".join(gk))
            levels.append(level_name)

    for i, dim in enumerate(dimensions):
        add_group((i,), dim)

    if include_pairs and d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                add_group((i, j), f"{dimensions[i]}_x_{dimensions[j]}")

    # bottom identity block, last
    for j, k in enumerate(keys):
        row = np.zeros(n_bottom)
        row[j] = 1.0
        rows.append(row)
        names.append("/".join(k))
        levels.append("bottom")

    S = np.vstack(rows)
    return Hierarchy(
        S=S,
        node_names=names,
        node_levels=levels,
        bottom_keys=list(keys),
        dimensions=list(dimensions),
    )


def shrink_covariance(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink a sample covariance toward its diagonal (Schafer-Strimmer).

    ``residuals`` has shape ``(T, n_nodes)``. With a wide hierarchy the sample
    covariance is rank-deficient (T < n_nodes is normal) and inverting it
    produces reconciliation weights that are noise. Shrinking the off-diagonal
    toward zero with the analytic intensity keeps MinT well-posed without a
    tuning parameter -- which is what makes MinT deployable rather than a paper
    result.

    Returns the shrunk covariance and the shrinkage intensity in [0, 1].
    """
    R = np.asarray(residuals, dtype=float)
    if R.ndim != 2:
        raise ValueError("residuals must be 2-D (T, n_nodes)")
    T, n = R.shape
    if T < 3:
        raise ValueError("need at least 3 residual observations")

    Rc = R - R.mean(axis=0, keepdims=True)
    S_hat = (Rc.T @ Rc) / (T - 1)
    d = np.sqrt(np.clip(np.diag(S_hat), 1e-12, None))
    corr = S_hat / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    # variance of each sample correlation, per Schafer & Strimmer eq. (2)
    Z = Rc / d
    Zbar = Z.mean(axis=0, keepdims=True)
    W = np.einsum("ti,tj->tij", Z - Zbar, Z - Zbar)
    var_r = W.var(axis=0, ddof=1) * T / ((T - 1) ** 2)

    off = ~np.eye(n, dtype=bool)
    denom = float(np.sum(corr[off] ** 2))
    numer = float(np.sum(var_r[off]))
    lam = 1.0 if denom < 1e-12 else float(np.clip(numer / denom, 0.0, 1.0))

    corr_shrunk = (1.0 - lam) * corr
    np.fill_diagonal(corr_shrunk, 1.0)
    W_shrunk = corr_shrunk * np.outer(d, d)
    # numerical symmetry + a ridge so the inverse is always defined
    W_shrunk = 0.5 * (W_shrunk + W_shrunk.T)
    W_shrunk += np.eye(n) * 1e-8 * float(np.mean(np.diag(W_shrunk)) + 1e-12)
    return W_shrunk, lam


def _G_bottom_up(S: np.ndarray) -> np.ndarray:
    n_nodes, n_bottom = S.shape
    G = np.zeros((n_bottom, n_nodes))
    G[:, n_nodes - n_bottom :] = np.eye(n_bottom)
    return G


def _G_top_down(S: np.ndarray, proportions: np.ndarray) -> np.ndarray:
    n_nodes, n_bottom = S.shape
    p = np.asarray(proportions, dtype=float).ravel()
    if p.size != n_bottom:
        raise ValueError("proportions must have one entry per bottom series")
    total = float(np.sum(p))
    p = np.full(n_bottom, 1.0 / n_bottom) if total <= 0 else p / total
    G = np.zeros((n_bottom, n_nodes))
    G[:, 0] = p  # row 0 of S is the grand total by construction
    return G


def _G_ols(S: np.ndarray) -> np.ndarray:
    return np.linalg.solve(S.T @ S, S.T)


def _G_mint(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    Wi = np.linalg.pinv(W)
    A = S.T @ Wi @ S
    return np.linalg.solve(A, S.T @ Wi)


def reconcile(
    base_forecasts: np.ndarray,
    hierarchy: Hierarchy,
    method: str = "mint",
    *,
    residuals: np.ndarray | None = None,
    proportions: np.ndarray | None = None,
    nonnegative: bool = True,
) -> np.ndarray:
    """Reconcile base forecasts for every node to a coherent set.

    Parameters
    ----------
    base_forecasts
        ``(n_nodes,)`` or ``(n_nodes, h)`` independent forecasts.
    method
        one of ``bottom_up``, ``top_down``, ``ols``, ``mint``.
    residuals
        ``(T, n_nodes)`` in-sample one-step residuals; required for ``mint``.
    proportions
        bottom-level shares; required for ``top_down``.
    nonnegative
        clip the reconciled bottom level at zero and re-aggregate. Demand is
        non-negative and a reconciliation that hands replenishment a negative
        cell is worse than an incoherent one.
    """
    S = hierarchy.S
    Y = np.asarray(base_forecasts, dtype=float)
    squeeze = Y.ndim == 1
    if squeeze:
        Y = Y[:, None]
    if Y.shape[0] != S.shape[0]:
        raise ValueError(
            f"expected {S.shape[0]} base forecast rows, got {Y.shape[0]}"
        )

    if method == "bottom_up":
        G = _G_bottom_up(S)
    elif method == "top_down":
        if proportions is None:
            raise ValueError("top_down requires historical proportions")
        G = _G_top_down(S, proportions)
    elif method == "ols":
        G = _G_ols(S)
    elif method == "mint":
        if residuals is None:
            raise ValueError("mint requires in-sample residuals")
        W, _ = shrink_covariance(residuals)
        G = _G_mint(S, W)
    else:
        raise ValueError(f"unknown reconciliation method: {method!r}")

    bottom = G @ Y
    if nonnegative:
        bottom = np.clip(bottom, 0.0, None)
    out = S @ bottom
    return out[:, 0] if squeeze else out


def coherency_error(values: np.ndarray, hierarchy: Hierarchy) -> float:
    """Max absolute violation of ``y = S @ y_bottom`` over all nodes."""
    Y = np.asarray(values, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n_bottom = hierarchy.n_bottom
    bottom = Y[Y.shape[0] - n_bottom :, :]
    return float(np.max(np.abs(hierarchy.S @ bottom - Y)))
