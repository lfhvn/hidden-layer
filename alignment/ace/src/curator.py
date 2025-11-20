"""
Curator component for ACE.

Synthesizes insights into context deltas and merges them deterministically.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from .context import Context, Strategy, Pitfall, ContextDelta, Insight, Example


class Curator:
    """
    Curator integrates insights into structured contexts.

    The curator synthesizes lessons into compact delta entries
    and merges them deterministically into the existing context.
    """

    def __init__(
        self,
        max_strategies_per_category: int = 10,
        max_pitfalls: int = 15,
        merge_strategy: str = "deterministic"
    ):
        """
        Initialize the Curator.

        Args:
            max_strategies_per_category: Maximum strategies per category
            max_pitfalls: Maximum total pitfalls
            merge_strategy: Strategy for merging deltas ("deterministic")
        """
        self.max_strategies_per_category = max_strategies_per_category
        self.max_pitfalls = max_pitfalls
        self.merge_strategy = merge_strategy

    def synthesize_delta(
        self,
        insights: List[Insight],
        context: Context
    ) -> ContextDelta:
        """
        Synthesize insights into a context delta.

        Args:
            insights: List of insights from reflection
            context: Current context

        Returns:
            ContextDelta to apply
        """
        delta_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Separate insights by type
        strategy_insights = [i for i in insights if i.type == "strategy"]
        pitfall_insights = [i for i in insights if i.type == "pitfall"]

        # Convert insights to strategies and pitfalls
        add_strategies = self._synthesize_strategies(strategy_insights, context)
        add_pitfalls = self._synthesize_pitfalls(pitfall_insights, context)

        # Determine modifications and removals
        modify_strategies = self._determine_modifications(add_strategies, context)
        remove_strategies = self._determine_removals(context)

        # Collect source trajectory IDs
        source_ids = set()
        for insight in insights:
            source_ids.update(insight.source_trajectory_ids)

        delta = ContextDelta(
            delta_id=delta_id,
            timestamp=timestamp,
            source_trajectory_ids=list(source_ids),
            insights=insights,
            add_strategies=add_strategies,
            add_pitfalls=add_pitfalls,
            modify_strategies=modify_strategies,
            remove_strategies=remove_strategies
        )

        return delta

    def merge_delta(
        self,
        context: Context,
        delta: ContextDelta
    ) -> Context:
        """
        Merge delta into context deterministically.

        This is a non-LLM, deterministic operation that ensures
        reproducible context evolution.

        Args:
            context: Current context
            delta: Delta to merge

        Returns:
            Updated context
        """
        # Create new context with incremented version
        new_context = Context(
            version=context.version + 1,
            domain=context.domain,
            base_prompt=context.base_prompt,
            strategies=context.strategies.copy(),
            pitfalls=context.pitfalls.copy(),
            history=context.history.copy(),
            metadata=context.metadata.copy(),
            last_updated=datetime.now()
        )

        # Apply removals first
        new_context.strategies = [
            s for s in new_context.strategies
            if s.id not in delta.remove_strategies
        ]
        new_context.pitfalls = [
            p for p in new_context.pitfalls
            if p.id not in delta.remove_pitfalls
        ]

        # Apply modifications
        for strategy_id, modifications in delta.modify_strategies.items():
            strategy = new_context.get_strategy(strategy_id)
            if strategy:
                for key, value in modifications.items():
                    if hasattr(strategy, key):
                        setattr(strategy, key, value)

        # Apply additions
        for strategy in delta.add_strategies:
            # Check if we already have a similar strategy
            existing = self._find_similar_strategy(strategy, new_context)
            if existing:
                # Merge with existing strategy
                self._merge_strategies(existing, strategy)
            else:
                # Add new strategy
                new_context.strategies.append(strategy)

        for pitfall in delta.add_pitfalls:
            # Check if we already have a similar pitfall
            existing = self._find_similar_pitfall(pitfall, new_context)
            if existing:
                # Merge with existing pitfall
                self._merge_pitfalls(existing, pitfall)
            else:
                # Add new pitfall
                new_context.pitfalls.append(pitfall)

        # Prune if needed (enforce limits)
        new_context = self._prune_context(new_context)

        # Add delta to history
        new_context.history.append({
            "delta_id": delta.delta_id,
            "timestamp": delta.timestamp.isoformat(),
            "num_insights": len(delta.insights),
            "added_strategies": len(delta.add_strategies),
            "added_pitfalls": len(delta.add_pitfalls),
            "modified_strategies": len(delta.modify_strategies),
            "removed_strategies": len(delta.remove_strategies)
        })

        return new_context

    def organize_strategies(self, context: Context) -> Context:
        """
        Reorganize strategies for better structure.

        Groups strategies by category and sorts by success rate.

        Args:
            context: Context to organize

        Returns:
            Organized context
        """
        # Group by category
        strategies_by_category = defaultdict(list)
        for strategy in context.strategies:
            strategies_by_category[strategy.category].append(strategy)

        # Sort each category by success rate
        organized_strategies = []
        for category in sorted(strategies_by_category.keys()):
            strategies = strategies_by_category[category]
            strategies.sort(key=lambda s: s.success_rate, reverse=True)
            organized_strategies.extend(strategies)

        # Create new context with organized strategies
        organized_context = Context(
            version=context.version,
            domain=context.domain,
            base_prompt=context.base_prompt,
            strategies=organized_strategies,
            pitfalls=context.pitfalls,
            history=context.history,
            metadata=context.metadata,
            last_updated=context.last_updated
        )

        return organized_context

    def _synthesize_strategies(
        self,
        insights: List[Insight],
        context: Context
    ) -> List[Strategy]:
        """Convert strategy insights to Strategy objects."""
        strategies = []

        for insight in insights:
            if not insight.description or not insight.category:
                continue

            strategy_id = str(uuid.uuid4())[:8]

            strategy = Strategy(
                id=f"strat_{strategy_id}",
                category=insight.category,
                description=insight.description,
                when_to_use=insight.when_to_use or "General use",
                examples=insight.examples,
                success_rate=insight.confidence,
                metadata={
                    "source": "ace_reflection",
                    "confidence": insight.confidence
                }
            )

            strategies.append(strategy)

        return strategies

    def _synthesize_pitfalls(
        self,
        insights: List[Insight],
        context: Context
    ) -> List[Pitfall]:
        """Convert pitfall insights to Pitfall objects."""
        pitfalls = []

        for insight in insights:
            if not insight.description or not insight.how_to_avoid:
                continue

            pitfall_id = str(uuid.uuid4())[:8]

            pitfall = Pitfall(
                id=f"pitfall_{pitfall_id}",
                description=insight.description,
                how_to_avoid=insight.how_to_avoid,
                frequency=insight.confidence,
                metadata={
                    "source": "ace_reflection",
                    "confidence": insight.confidence
                }
            )

            pitfalls.append(pitfall)

        return pitfalls

    def _determine_modifications(
        self,
        new_strategies: List[Strategy],
        context: Context
    ) -> Dict[str, Dict[str, Any]]:
        """Determine which existing strategies should be modified."""
        modifications = {}

        # For now, we don't modify existing strategies
        # In a more advanced implementation, we could:
        # - Update success rates based on new evidence
        # - Add examples to existing strategies
        # - Refine when_to_use conditions

        return modifications

    def _determine_removals(self, context: Context) -> List[str]:
        """Determine which strategies should be removed."""
        removals = []

        # Remove strategies with very low success rates
        for strategy in context.strategies:
            if strategy.success_rate < 0.2:
                removals.append(strategy.id)

        return removals

    def _find_similar_strategy(
        self,
        strategy: Strategy,
        context: Context
    ) -> Optional[Strategy]:
        """Find a similar existing strategy."""
        for existing in context.strategies:
            # Check if in same category and similar description
            if existing.category == strategy.category:
                # Simple similarity check (could be improved with embeddings)
                desc1 = existing.description.lower()
                desc2 = strategy.description.lower()

                # Check for significant word overlap
                words1 = set(desc1.split())
                words2 = set(desc2.split())
                overlap = len(words1 & words2)

                if overlap >= min(len(words1), len(words2)) * 0.5:
                    return existing

        return None

    def _find_similar_pitfall(
        self,
        pitfall: Pitfall,
        context: Context
    ) -> Optional[Pitfall]:
        """Find a similar existing pitfall."""
        for existing in context.pitfalls:
            # Simple similarity check
            desc1 = existing.description.lower()
            desc2 = pitfall.description.lower()

            words1 = set(desc1.split())
            words2 = set(desc2.split())
            overlap = len(words1 & words2)

            if overlap >= min(len(words1), len(words2)) * 0.5:
                return existing

        return None

    def _merge_strategies(self, existing: Strategy, new: Strategy):
        """Merge new strategy into existing strategy."""
        # Update success rate (weighted average)
        total_examples = len(existing.examples) + len(new.examples)
        if total_examples > 0:
            weight_existing = len(existing.examples) / total_examples
            weight_new = len(new.examples) / total_examples
            existing.success_rate = (
                existing.success_rate * weight_existing +
                new.success_rate * weight_new
            )

        # Add new examples (limit to 5 total)
        for example in new.examples:
            if len(existing.examples) < 5:
                existing.examples.append(example)

        # Update when_to_use if new one is more specific
        if len(new.when_to_use) > len(existing.when_to_use):
            existing.when_to_use = new.when_to_use

    def _merge_pitfalls(self, existing: Pitfall, new: Pitfall):
        """Merge new pitfall into existing pitfall."""
        # Update frequency (average)
        existing.frequency = (existing.frequency + new.frequency) / 2

        # Update how_to_avoid if new one is more detailed
        if len(new.how_to_avoid) > len(existing.how_to_avoid):
            existing.how_to_avoid = new.how_to_avoid

    def _prune_context(self, context: Context) -> Context:
        """Prune context to stay within limits."""
        # Prune strategies per category
        strategies_by_category = defaultdict(list)
        for strategy in context.strategies:
            strategies_by_category[strategy.category].append(strategy)

        pruned_strategies = []
        for category, strategies in strategies_by_category.items():
            # Sort by success rate and keep top N
            strategies.sort(key=lambda s: s.success_rate, reverse=True)
            pruned_strategies.extend(
                strategies[:self.max_strategies_per_category]
            )

        # Prune pitfalls
        pitfalls = context.pitfalls.copy()
        pitfalls.sort(key=lambda p: p.frequency, reverse=True)
        pruned_pitfalls = pitfalls[:self.max_pitfalls]

        context.strategies = pruned_strategies
        context.pitfalls = pruned_pitfalls

        return context
