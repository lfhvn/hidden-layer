"""
Introspection Research Project

Model introspection experiments inspired by Anthropic's recent findings.

Includes:
- Activation steering
- Concept vectors
- Introspection tasks
- API-based introspection (for frontier models)

Uses the harness for LLM provider abstraction and experiment tracking.
"""

# Introspection modules (optional - only if MLX is available)
try:
    from .activation_steering import ActivationCache, ActivationSteerer, SteeringConfig
    from .concept_vectors import (
        ConceptLibrary,
        ConceptVector,
        build_emotion_library,
        build_topic_library,
    )
    from .introspection_tasks import (
        IntrospectionEvaluator,
        IntrospectionResult,
        IntrospectionTask,
        IntrospectionTaskGenerator,
        IntrospectionTaskType,
    )

    _has_introspection = True
except ImportError:
    _has_introspection = False

# API Introspection (always available - doesn't require MLX)
try:
    from .introspection_api import (
        NATURAL_INTROSPECTION_PROMPTS,
        APIIntrospectionTester,
        PromptSteerer,
        PromptSteeringConfig,
    )

    _has_api_introspection = True
except ImportError:
    _has_api_introspection = False

__version__ = "0.1.0"

__all__ = []

# Add introspection exports if available
if _has_introspection:
    __all__.extend(
        [
            # Activation Steering
            "ActivationSteerer",
            "SteeringConfig",
            "ActivationCache",
            # Concept Vectors
            "ConceptLibrary",
            "ConceptVector",
            "build_emotion_library",
            "build_topic_library",
            # Introspection Tasks
            "IntrospectionTask",
            "IntrospectionResult",
            "IntrospectionTaskType",
            "IntrospectionTaskGenerator",
            "IntrospectionEvaluator",
        ]
    )

# Add API introspection exports if available
if _has_api_introspection:
    __all__.extend(
        [
            # API Introspection
            "APIIntrospectionTester",
            "PromptSteerer",
            "PromptSteeringConfig",
            "NATURAL_INTROSPECTION_PROMPTS",
        ]
    )
