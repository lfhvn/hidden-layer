"""
GeoMAS - Geometric Memory Analysis for Multi-Agent Systems

Analyze geometric vs associative memory structures in LLM systems.
"""

from .geometric_probes import (
    GeometricProbe,
    geometric_quality_score,
    compute_spectral_structure,
    visualize_geometry
)

from .multi_agent_analyzer import (
    MultiAgentGeometricAnalyzer,
    compare_strategies_geometrically,
    predict_strategy_benefit
)

__version__ = "0.1.0"
__all__ = [
    "GeometricProbe",
    "geometric_quality_score",
    "compute_spectral_structure",
    "visualize_geometry",
    "MultiAgentGeometricAnalyzer",
    "compare_strategies_geometrically",
    "predict_strategy_benefit",
]
