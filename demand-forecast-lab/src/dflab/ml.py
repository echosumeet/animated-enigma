"""Feature-based global forecaster on gradient-boosted trees.

This is a *cross-learning direct* forecaster: one model is fitted on every
series at once, the horizon is an input feature, and the target is
``y[t + h]``. Two design choices carry most of the weight.

**No leakage, enforced structurally.** Every feature for origin ``t`` is a
function of ``y[..t]`` only. Horizon-dependent features (the aligned seasonal
lag ``y[t + h - m]``, the target period's calendar position, the target
period's promotion flag) are built explicitly and are either in the past or
genuinely known in advance. The training origins are capped so that
``t + h <= train_end - 1``. There is no ``.shift()`` on a full-history frame
anywhere in this file, because that is where leakage usually gets in.

**Known-future covariates are treated as known.** A promotion calendar is
committed weeks ahead of execution, so the target period's promo flag is a
legitimate feature. Pretending otherwise understates what a feature model can
do; using an *unknown* future covariate the same way would overstate it. That
distinction is the difference between a model that works in the planning cycle
and one that only works in a notebook.

**Loss choice is a modelling decision, not a default.** Squared error fits the
conditional mean; absolute error fits the conditional median. On a series where
most periods are zero the conditional median *is* zero, so an absolute-error
model degenerates to forecasting nothing -- excellent MAE, useless for
replenishment. The benchmark reports both so the tradeoff is visible.

References
----------
Januschowski, T. et al. (2022). "Forecasting with trees." *International
Journal of Forecasting* 38(4), 1473-1481.

Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2022). "M5 accuracy
competition: results, findings and conclusions." *IJF* 38(4), 1346-1364.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .base import as_float_2d

__all__ = ["FeatureConfig", "build_features", "GlobalGBTForecaster"]


@dataclass
class FeatureConfig:
    """Which features get built. Defaults are tuned for weekly demand."""

    season_length: int = 52
    lags: tuple[int, ...] = (0, 1, 2, 3, 4, 8, 12, 26)
    roll_means: tuple[int, ...] = (4, 8, 13, 26, 52)
    roll_stds: tuple[int, ...] = (8, 26)
    zero_shares: tuple[int, ...] = (13, 52)
    nonzero_mean_window: int = 26
    use_seasonal_lag: bool = True
    use_promo: bool = True
    calendar_harmonics: int = 2
    static_dims: tuple[str, ...] = ("product", "region", "channel")

    def feature_names(self) -> list[str]:
        names = [f"lag_{k}" for k in self.lags]
        names += [f"rmean_{w}" for w in self.roll_means]
        names += [f"rstd_{w}" for w in self.roll_stds]
        names += [f"zshare_{w}" for w in self.zero_shares]
        names += [
            f"nzmean_{self.nonzero_mean_window}",
            "periods_since_demand",
            "adi_to_date",
            "cv2_to_date",
            "trend_slope_13",
        ]
        if self.use_promo:
            names += ["promo_t", "promo_count_4", "promo_target"]
        names += ["horizon"]
        if self.use_seasonal_lag:
            names += ["seasonal_lag", "seasonal_lag_ratio"]
        for k in range(1, self.calendar_harmonics + 1):
            names += [f"sin_{k}", f"cos_{k}"]
        names += ["week_of_year", "periods_of_history"]
        names += [f"static_{d}" for d in self.static_dims]
        return names

    def categorical_mask(self) -> np.ndarray:
        names = self.feature_names()
        return np.array([n.startswith("static_") for n in names], dtype=bool)


def _rolling(y: np.ndarray, w: int, fn) -> np.ndarray:
    """Causal rolling statistic: ``out[t]`` uses ``y[max(0,t-w+1)..t]``."""
    T = y.size
    out = np.empty(T)
    for t in range(T):
        lo = max(0, t - w + 1)
        out[t] = fn(y[lo : t + 1])
    return out


def _series_state(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Origin-indexed feature block for one series, shape (T, n_state)."""
    T = y.size
    cols: list[np.ndarray] = []

    for k in cfg.lags:
        col = np.zeros(T)
        if k == 0:
            col = y.copy()
        elif k < T:
            col[k:] = y[:-k]
        cols.append(col)

    for w in cfg.roll_means:
        cols.append(_rolling(y, w, np.mean))
    for w in cfg.roll_stds:
        cols.append(_rolling(y, w, lambda a: float(np.std(a)) if a.size > 1 else 0.0))
    for w in cfg.zero_shares:
        cols.append(_rolling(y, w, lambda a: float(np.mean(a <= 0))))

    w = cfg.nonzero_mean_window
    cols.append(
        _rolling(
            y,
            w,
            lambda a: float(np.mean(a[a > 0])) if np.any(a > 0) else 0.0,
        )
    )

    # periods since the last non-zero demand (the obsolescence signal)
    since = np.empty(T)
    counter = 0.0
    for t in range(T):
        counter = 0.0 if y[t] > 0 else counter + 1.0
        since[t] = counter
    cols.append(since)

    # running ADI and CV^2, computed only from history up to t
    adi_col = np.empty(T)
    cv2_col = np.empty(T)
    n_nz = 0
    s1 = 0.0
    s2 = 0.0
    for t in range(T):
        if y[t] > 0:
            n_nz += 1
            s1 += y[t]
            s2 += y[t] * y[t]
        adi_col[t] = (t + 1) / n_nz if n_nz else float(t + 1)
        if n_nz >= 2:
            mu = s1 / n_nz
            var = max(s2 / n_nz - mu * mu, 0.0)
            cv2_col[t] = var / (mu * mu) if mu > 0 else 0.0
        else:
            cv2_col[t] = 0.0
    cols.append(np.clip(adi_col, 0.0, 100.0))
    cols.append(np.clip(cv2_col, 0.0, 50.0))

    # short-run slope: recent 13-period mean vs the 13 before it
    slope = np.zeros(T)
    for t in range(T):
        a = y[max(0, t - 12) : t + 1]
        b = y[max(0, t - 25) : max(0, t - 12)]
        if b.size:
            slope[t] = float(np.mean(a) - np.mean(b))
    cols.append(slope)

    return np.column_stack(cols)


def build_features(
    panel,
    origins,
    horizons,
    cfg: FeatureConfig,
    *,
    promo=None,
    static=None,
    targets=True,
):
    """Assemble the design matrix for a set of (series, origin, horizon) points.

    Parameters
    ----------
    panel
        ``(n_series, T)`` history. Only ``panel[:, :max(origins)+1]`` is read
        for feature construction; targets are read at ``origin + horizon``.
    origins
        1-D array of origin indices ``t`` (features use data through ``t``).
    horizons
        1-D array of horizons ``h >= 1``.
    promo
        optional ``(n_series, T_full)`` binary promotion flags; must extend to
        ``max(origin) + max(horizon)`` because promo is a known-future
        covariate.
    static
        optional ``(n_series, n_static)`` integer-coded categorical features.
    targets
        when True also return ``y`` at ``origin + horizon``.

    Returns
    -------
    (X, y, index) where ``index`` is an ``(n_rows, 3)`` int array of
    ``(series, origin, horizon)``.
    """
    P = as_float_2d(panel)
    n_series, T = P.shape
    O = np.asarray(origins, dtype=int).ravel()
    H = np.asarray(horizons, dtype=int).ravel()
    if O.size == 0 or H.size == 0:
        raise ValueError("need at least one origin and one horizon")
    if np.any(H < 1):
        raise ValueError("horizons must be >= 1")
    max_needed = int(O.max() + H.max())
    if targets and max_needed >= T:
        raise ValueError(
            f"origin+horizon reaches index {max_needed} but panel has {T} periods"
        )

    m = cfg.season_length
    if promo is not None:
        PR = np.asarray(promo, dtype=float)
        if PR.shape[0] != n_series or PR.shape[1] <= max_needed:
            raise ValueError("promo must cover every series through origin+horizon")
    else:
        PR = None

    if static is not None:
        ST = np.asarray(static, dtype=float)
        if ST.shape[0] != n_series:
            raise ValueError("static must have one row per series")
    else:
        ST = np.zeros((n_series, len(cfg.static_dims)))

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    idx_blocks: list[np.ndarray] = []

    for i in range(n_series):
        state = _series_state(P[i], cfg)  # (T, n_state)
        base = state[O]  # (n_origins, n_state)
        for h in H:
            tgt = O + int(h)
            block = [base]
            extra: list[np.ndarray] = []

            if cfg.use_promo:
                p_t = PR[i][O] if PR is not None else np.zeros(O.size)
                p_c4 = (
                    np.array([PR[i][max(0, t - 3) : t + 1].sum() for t in O])
                    if PR is not None
                    else np.zeros(O.size)
                )
                p_tg = PR[i][tgt] if PR is not None else np.zeros(O.size)
                extra += [p_t, p_c4, p_tg]

            extra.append(np.full(O.size, float(h)))

            if cfg.use_seasonal_lag:
                slag_idx = tgt - m
                valid = slag_idx >= 0
                slag = np.zeros(O.size)
                slag[valid] = P[i][slag_idx[valid]]
                denom = np.maximum(base[:, len(cfg.lags)], 1e-6)  # rmean_4 column
                extra += [slag, slag / denom]

            woy = tgt % m
            for k in range(1, cfg.calendar_harmonics + 1):
                extra.append(np.sin(2 * np.pi * k * woy / m))
                extra.append(np.cos(2 * np.pi * k * woy / m))
            extra.append(woy.astype(float))
            extra.append(O.astype(float) + 1.0)

            for j in range(ST.shape[1]):
                extra.append(np.full(O.size, ST[i, j]))

            block.append(np.column_stack(extra))
            X_blocks.append(np.hstack(block))
            if targets:
                y_blocks.append(P[i][tgt])
            idx_blocks.append(
                np.column_stack(
                    [np.full(O.size, i), O, np.full(O.size, int(h))]
                ).astype(int)
            )

    X = np.vstack(X_blocks)
    index = np.vstack(idx_blocks)
    y = np.concatenate(y_blocks) if targets else None
    return X, y, index


class GlobalGBTForecaster:
    """Cross-learning direct forecaster on ``HistGradientBoostingRegressor``.

    Fitted with ``fit_panel(panel)`` on the training window only; the origin
    cap guarantees no target is drawn from beyond ``panel.shape[1] - 1``.
    ``predict_panel(h)`` returns a ``(n_series, h)`` matrix forecast from the
    final origin.
    """

    is_global = True

    def __init__(
        self,
        horizon: int = 13,
        cfg: FeatureConfig | None = None,
        *,
        loss: str = "squared_error",
        quantile: float | None = None,
        max_iter: int = 220,
        learning_rate: float = 0.06,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        max_origins: int = 130,
        random_state: int = 0,
        name: str | None = None,
        promo: np.ndarray | None = None,
        static: np.ndarray | None = None,
    ) -> None:
        self.horizon = int(horizon)
        self.cfg = cfg or FeatureConfig()
        self.loss = loss
        self.quantile = quantile
        self.max_origins = int(max_origins)
        self.promo = promo
        self.static = static
        self._model_kwargs = dict(
            loss=loss,
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
            early_stopping=False,
        )
        if quantile is not None:
            self._model_kwargs["quantile"] = float(quantile)
        if name is not None:
            self.name = name
        elif quantile is not None:
            self.name = f"gbt_q{quantile:g}"
        else:
            self.name = f"gbt_{loss.split('_')[0]}"
        self.model_: HistGradientBoostingRegressor | None = None
        self._panel: np.ndarray | None = None
        self.n_train_rows_ = 0

    # -- fitting -----------------------------------------------------------
    def _training_origins(self, T: int) -> np.ndarray:
        h_max = self.horizon
        m = self.cfg.season_length
        first = max(m // 2, min(m, T // 3))
        last = T - 1 - h_max
        if last <= first:
            first = max(4, last - 1)
        if last <= first:
            raise ValueError(
                f"training window of {T} periods is too short for horizon {h_max}"
            )
        origins = np.arange(first, last + 1)
        if origins.size > self.max_origins:
            origins = origins[-self.max_origins :]
        return origins

    def fit_panel(self, panel) -> "GlobalGBTForecaster":
        P = as_float_2d(panel)
        self._panel = P
        T = P.shape[1]
        origins = self._training_origins(T)
        horizons = np.arange(1, self.horizon + 1)
        X, y, _ = build_features(
            P,
            origins,
            horizons,
            self.cfg,
            promo=self.promo,
            static=self.static,
            targets=True,
        )
        self.n_train_rows_ = int(X.shape[0])
        cat = self.cfg.categorical_mask()
        model = HistGradientBoostingRegressor(
            categorical_features=cat if cat.any() else None, **self._model_kwargs
        )
        if self.loss in ("poisson", "gamma"):
            y = np.clip(y, 1e-6 if self.loss == "gamma" else 0.0, None)
        model.fit(X, y)
        self.model_ = model
        return self

    # -- prediction --------------------------------------------------------
    def predict_panel(self, h: int) -> np.ndarray:
        if self.model_ is None or self._panel is None:
            raise RuntimeError(f"{self.name}: predict_panel() before fit_panel()")
        h = int(h)
        if h < 1:
            raise ValueError("horizon must be >= 1")
        if h > self.horizon:
            raise ValueError(
                f"{self.name} was fitted for horizon {self.horizon}, asked for {h}"
            )
        P = self._panel
        n_series, T = P.shape
        origins = np.array([T - 1], dtype=int)
        horizons = np.arange(1, h + 1)
        X, _, index = build_features(
            P,
            origins,
            horizons,
            self.cfg,
            promo=self.promo,
            static=self.static,
            targets=False,
        )
        preds = np.clip(self.model_.predict(X), 0.0, None)
        out = np.zeros((n_series, h))
        out[index[:, 0], index[:, 2] - 1] = preds
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<GlobalGBTForecaster name={self.name!r} horizon={self.horizon}>"
