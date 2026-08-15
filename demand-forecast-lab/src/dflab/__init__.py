"""dflab -- a benchmarking harness for intermittent and hierarchical demand forecasting.

The point of this package is not to provide another forecasting API. It is to
answer one question properly: **which forecasting method wins where**, split by
demand pattern rather than averaged into a single misleading number, and
measured with metrics that survive a portfolio full of zeros.

Layout
------
``datagen``       reproducible product x region x channel demand generator
``classify``      ADI / CV^2 demand classification (Syntetos-Boylan-Croston)
``baselines``     naive, seasonal naive, moving average, drift, mean, zero
``ets``           SES, Holt, Holt-Winters (recursions + scipy parameter fit)
``intermittent``  Croston, SBA, TSB
``ml``            global feature-based gradient boosting, leakage-safe
``hierarchy``     summing matrix, bottom-up / top-down / OLS / MinT
``metrics``       WAPE, MASE, sMAPE, RMSSE, bias, tracking signal, pinball
``backtest``      rolling-origin evaluation with per-quadrant aggregation
``pipeline``      the standard model zoo and reconciliation study
``plots``         the figures used in the README
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Sumeet"

from .backtest import BacktestConfig, BacktestResult, backtest, rolling_origins
from .baselines import (
    DriftForecaster,
    MeanForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ZeroForecaster,
)
from .classify import DemandProfile, adi, classify_panel, classify_series, cv_squared
from .datagen import DGPConfig, DemandPanel, generate_panel
from .ets import HoltLinear, HoltWinters, SimpleExponentialSmoothing
from .hierarchy import (
    Hierarchy,
    build_hierarchy,
    coherency_error,
    reconcile,
    shrink_covariance,
)
from .intermittent import CrostonForecaster, SBAForecaster, TSBForecaster
from .metrics import (
    mase,
    pinball_loss,
    rmsse,
    smape,
    tracking_signal,
    wape,
)
from .ml import FeatureConfig, GlobalGBTForecaster, build_features
from .pipeline import (
    build_model_zoo,
    run_backtest,
    run_reconciliation_study,
)

__all__ = [
    "__version__",
    "BacktestConfig",
    "BacktestResult",
    "backtest",
    "rolling_origins",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "DriftForecaster",
    "MeanForecaster",
    "ZeroForecaster",
    "SimpleExponentialSmoothing",
    "HoltLinear",
    "HoltWinters",
    "CrostonForecaster",
    "SBAForecaster",
    "TSBForecaster",
    "FeatureConfig",
    "GlobalGBTForecaster",
    "build_features",
    "Hierarchy",
    "build_hierarchy",
    "reconcile",
    "shrink_covariance",
    "coherency_error",
    "DGPConfig",
    "DemandPanel",
    "generate_panel",
    "DemandProfile",
    "classify_series",
    "classify_panel",
    "adi",
    "cv_squared",
    "wape",
    "mase",
    "rmsse",
    "smape",
    "pinball_loss",
    "tracking_signal",
    "build_model_zoo",
    "run_backtest",
    "run_reconciliation_study",
]
