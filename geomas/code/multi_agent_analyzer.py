"""
Multi-agent geometric analysis tools.

Compare geometric structures across different reasoning strategies
(single, debate, manager-worker, self-consistency).
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from .geometric_probes import GeometricProbe, GeometricAnalysis

try:
    from harness import run_strategy, StrategyResult
    HARNESS_AVAILABLE = True
except ImportError:
    HARNESS_AVAILABLE = False
    StrategyResult = None


@dataclass
class StrategyGeometricComparison:
    """Results of comparing geometric structures across strategies"""
    strategy_name: str
    geometric_analysis: GeometricAnalysis
    task_performance: Optional[float] = None
    latency_s: Optional[float] = None
    cost_usd: Optional[float] = None


@dataclass
class MultiAgentGeometricResult:
    """Full results of multi-agent geometric analysis"""
    task_input: str
    comparisons: List[StrategyGeometricComparison]
    recommendation: str
    confidence: float
    metadata: Dict[str, Any]


class MultiAgentGeometricAnalyzer:
    """
    Analyze and compare geometric structures across multi-agent strategies.

    This class provides tools to:
    1. Run multiple strategies on the same task
    2. Extract and compare geometric structures
    3. Predict which strategy will perform best
    4. Visualize differences in geometric organization
    """

    def __init__(
        self,
        model: str,
        provider: str = "ollama",
        layer_indices: Optional[List[int]] = None
    ):
        """
        Initialize multi-agent geometric analyzer.

        Args:
            model: Model name/path
            provider: LLM provider
            layer_indices: Which layers to analyze
        """
        self.model = model
        self.provider = provider
        self.probe = GeometricProbe(model, provider, layer_indices)

    def compare_strategies_geometrically(
        self,
        task_input: str,
        strategies: List[str] = None,
        extract_hidden_states_fn: Optional[callable] = None,
        **strategy_kwargs
    ) -> MultiAgentGeometricResult:
        """
        Run multiple strategies and compare their geometric structures.

        Args:
            task_input: The task to analyze
            strategies: List of strategy names (default: ["single", "debate"])
            extract_hidden_states_fn: Custom function to extract hidden states
            **strategy_kwargs: Additional arguments for strategies

        Returns:
            MultiAgentGeometricResult with comparisons and recommendation
        """
        if strategies is None:
            strategies = ["single", "debate"]

        if not HARNESS_AVAILABLE:
            raise ImportError(
                "Harness not available. Ensure you're in the hidden-layer project."
            )

        comparisons = []

        for strategy in strategies:
            # Run strategy
            result = run_strategy(
                strategy,
                task_input=task_input,
                provider=self.provider,
                model=self.model,
                **strategy_kwargs
            )

            # Extract hidden states
            # TODO: Implement actual extraction from model
            # For now, this is a placeholder
            if extract_hidden_states_fn:
                hidden_states = extract_hidden_states_fn(result)
            else:
                # Placeholder: would need real extraction
                hidden_states = self._placeholder_extraction(result)

            # Analyze geometry
            if hidden_states is not None:
                analysis = self.probe.analyze(hidden_states)
            else:
                # Create dummy analysis if extraction failed
                analysis = self._create_dummy_analysis()

            # Record comparison
            comparison = StrategyGeometricComparison(
                strategy_name=strategy,
                geometric_analysis=analysis,
                task_performance=None,  # Would need evaluation
                latency_s=result.latency_s if hasattr(result, 'latency_s') else None,
                cost_usd=result.cost_usd if hasattr(result, 'cost_usd') else None
            )
            comparisons.append(comparison)

        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(comparisons)

        return MultiAgentGeometricResult(
            task_input=task_input,
            comparisons=comparisons,
            recommendation=recommendation,
            confidence=confidence,
            metadata={
                'model': self.model,
                'provider': self.provider,
                'strategies_compared': strategies
            }
        )

    def _placeholder_extraction(self, result: Any) -> Optional[np.ndarray]:
        """Placeholder for hidden state extraction"""
        # In real implementation, would extract from model
        # For now, return None to indicate not implemented
        return None

    def _create_dummy_analysis(self) -> GeometricAnalysis:
        """Create dummy analysis for when extraction fails"""
        return GeometricAnalysis(
            spectral_gap=0.0,
            fiedler_vector=np.array([]),
            eigenvalues=np.array([]),
            eigenvectors=np.array([[]]),
            cluster_coherence=0.5,
            global_structure_score=0.5,
            quality_score=0.5,
            metadata={'placeholder': True}
        )

    def _generate_recommendation(
        self,
        comparisons: List[StrategyGeometricComparison]
    ) -> tuple[str, float]:
        """
        Generate strategy recommendation based on geometric analysis.

        Logic:
        - If single-model geometric quality is high (>0.7), recommend single
        - If low (<0.5), recommend multi-agent (debate or manager-worker)
        - If medium, consider cost-benefit tradeoff
        """
        single_comparison = next(
            (c for c in comparisons if c.strategy_name == "single"),
            None
        )

        if single_comparison is None:
            return "debate", 0.5  # Default if no single baseline

        single_quality = single_comparison.geometric_analysis.quality_score

        if single_quality > 0.7:
            return "single", 0.8  # High confidence in single model
        elif single_quality < 0.5:
            # Low quality → recommend multi-agent
            # Choose between debate and manager-worker based on task
            return "debate", 0.7
        else:
            # Medium quality → uncertain, slight preference for multi-agent
            return "debate", 0.5

    def geometric_evolution_across_rounds(
        self,
        debate_result: Any,
        extract_per_round_fn: Optional[callable] = None
    ) -> List[GeometricAnalysis]:
        """
        Analyze how geometric structure evolves across debate rounds.

        Args:
            debate_result: Result from debate strategy
            extract_per_round_fn: Function to extract hidden states per round

        Returns:
            List of GeometricAnalysis for each round
        """
        if extract_per_round_fn is None:
            raise ValueError("Must provide extract_per_round_fn for round analysis")

        evolution = []

        # Extract hidden states for each round
        per_round_states = extract_per_round_fn(debate_result)

        for round_idx, hidden_states in enumerate(per_round_states):
            analysis = self.probe.analyze(hidden_states)
            evolution.append(analysis)

        return evolution

    def predict_strategy_benefit(
        self,
        task_input: str,
        baseline_strategy: str = "single",
        candidate_strategies: List[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Predict which multi-agent strategies will benefit from geometric analysis.

        Args:
            task_input: The task to analyze
            baseline_strategy: Baseline strategy (usually "single")
            candidate_strategies: Strategies to compare against baseline

        Returns:
            Dictionary mapping strategy name to predicted benefit score
        """
        if candidate_strategies is None:
            candidate_strategies = ["debate", "manager_worker", "self_consistency"]

        # Run baseline
        baseline_result = run_strategy(
            baseline_strategy,
            task_input=task_input,
            provider=self.provider,
            model=self.model,
            **kwargs
        )

        # Extract and analyze baseline geometry
        baseline_states = self._placeholder_extraction(baseline_result)
        if baseline_states is not None:
            baseline_analysis = self.probe.analyze(baseline_states)
            baseline_quality = baseline_analysis.quality_score
        else:
            baseline_quality = 0.5  # Placeholder

        # Predict benefit for each candidate strategy
        predictions = {}

        for strategy in candidate_strategies:
            # Simple heuristic: benefit inversely proportional to baseline quality
            if baseline_quality < 0.4:
                predicted_benefit = 0.7 + (0.4 - baseline_quality)  # High benefit
            elif baseline_quality < 0.6:
                predicted_benefit = 0.5  # Medium benefit
            else:
                predicted_benefit = 0.3 - (baseline_quality - 0.6)  # Low benefit

            # Clip to [0, 1]
            predicted_benefit = max(0.0, min(1.0, predicted_benefit))

            predictions[strategy] = predicted_benefit

        return predictions

    def visualize_comparison(
        self,
        comparison_result: MultiAgentGeometricResult,
        save_path: Optional[str] = None
    ) -> Any:
        """
        Visualize geometric differences across strategies.

        Args:
            comparison_result: Result from compare_strategies_geometrically
            save_path: Optional path to save visualization

        Returns:
            Figure object (or None if plotly not available)
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create subplots for each strategy
            n_strategies = len(comparison_result.comparisons)

            fig = make_subplots(
                rows=1,
                cols=n_strategies,
                subplot_titles=[c.strategy_name for c in comparison_result.comparisons]
            )

            for idx, comparison in enumerate(comparison_result.comparisons):
                analysis = comparison.geometric_analysis

                # Plot eigenvalue spectrum
                trace = go.Bar(
                    x=list(range(len(analysis.eigenvalues))),
                    y=analysis.eigenvalues,
                    name=comparison.strategy_name,
                    showlegend=False
                )
                fig.add_trace(trace, row=1, col=idx+1)

                # Add quality score annotation
                fig.add_annotation(
                    text=f"Quality: {analysis.quality_score:.3f}",
                    xref=f"x{idx+1}",
                    yref=f"y{idx+1}",
                    x=len(analysis.eigenvalues) / 2,
                    y=max(analysis.eigenvalues) * 0.9,
                    showarrow=False
                )

            fig.update_layout(
                title=f"Geometric Structure Comparison<br><sub>Task: {comparison_result.task_input[:50]}...</sub>",
                template='plotly_white',
                height=400
            )

            if save_path:
                fig.write_html(save_path)

            return fig

        except ImportError:
            print("Plotly not available for visualization")
            return None


# Standalone convenience functions

def compare_strategies_geometrically(
    task_input: str,
    model: str,
    provider: str = "ollama",
    strategies: List[str] = None,
    **kwargs
) -> MultiAgentGeometricResult:
    """Standalone function for strategy comparison"""
    analyzer = MultiAgentGeometricAnalyzer(model, provider)
    return analyzer.compare_strategies_geometrically(task_input, strategies, **kwargs)


def predict_strategy_benefit(
    task_input: str,
    model: str,
    provider: str = "ollama",
    **kwargs
) -> Dict[str, float]:
    """Standalone function for benefit prediction"""
    analyzer = MultiAgentGeometricAnalyzer(model, provider)
    return analyzer.predict_strategy_benefit(task_input, **kwargs)
