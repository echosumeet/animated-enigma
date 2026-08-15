"""etarisk: ETA prediction, conformal intervals and delay-risk decisions.

A small, honest pipeline over a synthetic freight network: generate shipments
with a heavy right tail, engineer leakage-safe features, fit a point ETA model,
wrap it in split-conformal intervals, calibrate a delay-risk classifier, and
turn the probability into an expedite/notify/nothing decision under an explicit
cost matrix.
"""

from .decision import ACTIONS, CostMatrix, compare_to_baseline, optimal_actions
from .drift import psi, psi_table
from .features import TargetEncoder, build_features, temporal_folds, temporal_split
from .generate import GeneratorConfig, default_network, describe_distribution, generate_shipments
from .model import ConformalETA, ETAModel, mae_by_horizon
from .pipeline import PipelineResult, run_pipeline
from .risk import DelayRiskModel, expected_calibration_error, reliability_table
from .rng import StreamBank

__version__ = "0.1.0"

__all__ = [
    "ACTIONS",
    "ConformalETA",
    "CostMatrix",
    "DelayRiskModel",
    "ETAModel",
    "GeneratorConfig",
    "PipelineResult",
    "StreamBank",
    "TargetEncoder",
    "build_features",
    "compare_to_baseline",
    "default_network",
    "describe_distribution",
    "expected_calibration_error",
    "generate_shipments",
    "mae_by_horizon",
    "optimal_actions",
    "psi",
    "psi_table",
    "reliability_table",
    "run_pipeline",
    "temporal_folds",
    "temporal_split",
    "__version__",
]
