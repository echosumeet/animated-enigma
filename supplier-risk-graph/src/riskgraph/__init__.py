"""riskgraph: multi-tier supplier risk analysis on a typed supply network graph.

The premise: companies know tier 1 and are blind past it, and the concentration risk
lives at tier 3.
"""

from .concentration import (
    HiddenDependency,
    effective_sources,
    geographic_hhi,
    hhi,
    hhi_by_tier,
    hidden_dependencies,
    supplier_hhi,
)
from .flow import ExpandedPart, expand, output_fraction, revenue_at_risk, site_flow, supplier_flow
from .generate import generate_network
from .mitigation import Mitigation, score_actions, with_dual_source, with_prequalification
from .model import (
    BomEdge,
    Part,
    Site,
    Supplier,
    SupplyEdge,
    SupplyNetwork,
    load_network,
)
from .simulate import SimResult, simulate
from .spof import Spof, articulation_points, degree_ranking, rank_spofs, sole_source_parts

__version__ = "0.1.0"

__all__ = [
    "BomEdge", "ExpandedPart", "HiddenDependency", "Mitigation", "Part", "SimResult", "Site",
    "Spof", "Supplier", "SupplyEdge", "SupplyNetwork", "articulation_points", "degree_ranking",
    "effective_sources", "expand", "generate_network", "geographic_hhi", "hhi", "hhi_by_tier",
    "hidden_dependencies", "load_network", "output_fraction", "rank_spofs",
    "revenue_at_risk", "score_actions", "simulate", "site_flow", "sole_source_parts",
    "supplier_flow", "supplier_hhi", "with_dual_source", "with_prequalification", "__version__",
]
