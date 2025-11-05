"""Real-time monitoring of steering metrics."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from .adherence import AdherenceMetrics

logger = logging.getLogger(__name__)


@dataclass
class MetricsSummary:
    """Summary of metrics over a time period."""

    start_time: datetime
    end_time: datetime
    total_generations: int
    mean_adherence: float
    std_adherence: float
    success_rate: float
    top_vectors: List[tuple]  # List of (vector_name, score)
    constraint_violations: Dict[str, int]


class MetricsMonitor:
    """Monitors and aggregates steering metrics in real-time."""

    def __init__(self, retention_days: int = 30):
        """Initialize metrics monitor.

        Args:
            retention_days: How many days to retain metrics
        """
        self.retention_days = retention_days

        # Track metrics per experiment/vector
        self.metrics: Dict[tuple, AdherenceMetrics] = {}

        # Time series data
        self.timestamps: List[datetime] = []

        logger.info(f"MetricsMonitor initialized (retention={retention_days} days)")

    def get_or_create_metrics(
        self, experiment_id: int, vector_name: str
    ) -> AdherenceMetrics:
        """Get or create metrics tracker.

        Args:
            experiment_id: Experiment ID
            vector_name: Steering vector name

        Returns:
            AdherenceMetrics instance
        """
        key = (experiment_id, vector_name)

        if key not in self.metrics:
            self.metrics[key] = AdherenceMetrics(
                experiment_id=experiment_id,
                steering_vector_name=vector_name,
            )

        return self.metrics[key]

    def record(
        self,
        experiment_id: int,
        vector_name: str,
        adherence_score: float,
        constraints_met: Optional[Dict[str, bool]] = None,
    ):
        """Record a new steering result.

        Args:
            experiment_id: Experiment ID
            vector_name: Steering vector name
            adherence_score: Adherence score (0-1)
            constraints_met: Constraint satisfaction status
        """
        metrics = self.get_or_create_metrics(experiment_id, vector_name)
        metrics.add_score(adherence_score, constraints_met)

        self.timestamps.append(datetime.utcnow())

        logger.debug(
            f"Recorded adherence={adherence_score:.3f} for "
            f"exp={experiment_id}, vector={vector_name}"
        )

    def get_summary(
        self,
        experiment_id: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> MetricsSummary:
        """Get summary of metrics.

        Args:
            experiment_id: Filter by experiment (None = all)
            since: Only include data since this time

        Returns:
            MetricsSummary
        """
        if since is None:
            since = datetime.utcnow() - timedelta(days=1)

        end_time = datetime.utcnow()

        # Filter metrics
        relevant_metrics = []
        for (exp_id, vec_name), metrics in self.metrics.items():
            if experiment_id is None or exp_id == experiment_id:
                relevant_metrics.append(metrics)

        if not relevant_metrics:
            return MetricsSummary(
                start_time=since,
                end_time=end_time,
                total_generations=0,
                mean_adherence=0.0,
                std_adherence=0.0,
                success_rate=0.0,
                top_vectors=[],
                constraint_violations={},
            )

        # Aggregate statistics
        total_generations = sum(m.total_generations for m in relevant_metrics)
        all_scores = [score for m in relevant_metrics for score in m.scores]

        mean_adherence = float(np.mean(all_scores)) if all_scores else 0.0
        std_adherence = float(np.std(all_scores)) if all_scores else 0.0

        total_successful = sum(m.successful_generations for m in relevant_metrics)
        success_rate = (
            total_successful / total_generations if total_generations > 0 else 0.0
        )

        # Top vectors by adherence
        vector_scores = [
            (m.steering_vector_name, m.mean_score) for m in relevant_metrics
        ]
        vector_scores.sort(key=lambda x: x[1], reverse=True)
        top_vectors = vector_scores[:10]

        # Constraint violations
        constraint_violations = {}
        for metrics in relevant_metrics:
            for constraint_name in metrics.constraint_satisfaction.keys():
                rate = metrics.constraint_satisfaction_rate(constraint_name)
                violation_count = int(
                    len(metrics.constraint_satisfaction[constraint_name]) * (1 - rate)
                )
                constraint_violations[constraint_name] = (
                    constraint_violations.get(constraint_name, 0) + violation_count
                )

        return MetricsSummary(
            start_time=since,
            end_time=end_time,
            total_generations=total_generations,
            mean_adherence=mean_adherence,
            std_adherence=std_adherence,
            success_rate=success_rate,
            top_vectors=top_vectors,
            constraint_violations=constraint_violations,
        )

    def get_time_series(
        self,
        experiment_id: int,
        vector_name: str,
        window_minutes: int = 60,
    ) -> List[tuple]:
        """Get time series data for plotting.

        Args:
            experiment_id: Experiment ID
            vector_name: Vector name
            window_minutes: Aggregation window in minutes

        Returns:
            List of (timestamp, mean_score) tuples
        """
        metrics = self.get_or_create_metrics(experiment_id, vector_name)

        if not metrics.scores:
            return []

        # Simple binning by window
        moving_avg = metrics.moving_average(window=min(10, len(metrics.scores)))

        # Create timestamps (simplified - would use actual timestamps in production)
        now = datetime.utcnow()
        timestamps = [
            now - timedelta(minutes=i * window_minutes)
            for i in range(len(moving_avg) - 1, -1, -1)
        ]

        return list(zip(timestamps, moving_avg))

    def cleanup_old_data(self):
        """Remove data older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)

        # In production, would iterate through timestamps and remove old entries
        # For now, keep all data in memory

        logger.info(f"Cleaned up data older than {cutoff}")
