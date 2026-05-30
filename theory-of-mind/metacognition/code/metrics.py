"""
Calibration metrics for metacognition experiments.

Pure-stdlib implementation (no numpy) so it runs anywhere - CI, ephemeral
containers, air-gapped machines. Every metric takes two parallel sequences:

    confidences: probability the model assigned to its own answer (in [0, 1])
    correct:     whether that answer was actually right (bool / 0-1)

These quantify *metacognitive calibration*: does a model's stated confidence
track whether it is actually right?

References:
  - Brier (1950), "Verification of forecasts expressed in terms of probability"
  - Guo et al. (2017), "On Calibration of Modern Neural Networks" (ECE/reliability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


def _validate(confidences: Sequence[float], correct: Sequence[bool]) -> Tuple[List[float], List[int]]:
    """Coerce, length-check, and range-check inputs. Returns (confidences, correct_as_int)."""
    conf = [float(c) for c in confidences]
    corr = [int(bool(c)) for c in correct]
    if len(conf) != len(corr):
        raise ValueError(f"Length mismatch: {len(conf)} confidences vs {len(corr)} labels.")
    if not conf:
        raise ValueError("Cannot compute calibration metrics on an empty sample.")
    for c in conf:
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"Confidence {c} outside [0, 1]. Convert percentages to fractions first.")
    return conf, corr


def accuracy(correct: Sequence[bool]) -> float:
    """Fraction of answers that were correct."""
    corr = [int(bool(c)) for c in correct]
    if not corr:
        raise ValueError("Cannot compute accuracy on an empty sample.")
    return sum(corr) / len(corr)


def mean_confidence(confidences: Sequence[float]) -> float:
    """Average stated confidence."""
    conf = [float(c) for c in confidences]
    if not conf:
        raise ValueError("Cannot compute mean confidence on an empty sample.")
    return sum(conf) / len(conf)


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """
    Mean squared error between confidence and outcome. Lower is better (0 = perfect).

    A *proper* scoring rule: it is minimized in expectation only by reporting your
    true probability of being correct, so it rewards honest confidence.
    """
    conf, corr = _validate(confidences, correct)
    return sum((c - o) ** 2 for c, o in zip(conf, corr)) / len(conf)


@dataclass
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lo: float
    hi: float
    count: int
    avg_confidence: float  # mean stated confidence of items in this bin
    accuracy: float  # fraction of items in this bin that were correct

    @property
    def gap(self) -> float:
        """Signed calibration gap (confidence - accuracy). Positive => overconfident."""
        return self.avg_confidence - self.accuracy


@dataclass
class CalibrationReport:
    """Full calibration summary for one set of (confidence, correctness) pairs."""

    n: int
    accuracy: float
    mean_confidence: float
    brier: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    overconfidence: float  # mean_confidence - accuracy (signed)
    bins: List[ReliabilityBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "mean_confidence": self.mean_confidence,
            "brier": self.brier,
            "ece": self.ece,
            "mce": self.mce,
            "overconfidence": self.overconfidence,
            "bins": [
                {
                    "range": [b.lo, b.hi],
                    "count": b.count,
                    "avg_confidence": b.avg_confidence,
                    "accuracy": b.accuracy,
                    "gap": b.gap,
                }
                for b in self.bins
            ],
        }


def reliability_bins(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> List[ReliabilityBin]:
    """
    Partition predictions into equal-width confidence bins and report, per bin, the
    average confidence vs. the empirical accuracy. Perfectly calibrated => the two are
    equal in every bin (points lie on the diagonal of a reliability diagram).

    Empty bins are omitted from the returned list.
    """
    conf, corr = _validate(confidences, correct)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: List[ReliabilityBin] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Last bin is inclusive of the upper edge so confidence == 1.0 lands somewhere.
        if b == n_bins - 1:
            members = [(c, o) for c, o in zip(conf, corr) if lo <= c <= hi]
        else:
            members = [(c, o) for c, o in zip(conf, corr) if lo <= c < hi]
        if not members:
            continue
        cs = [c for c, _ in members]
        os_ = [o for _, o in members]
        bins.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                count=len(members),
                avg_confidence=sum(cs) / len(cs),
                accuracy=sum(os_) / len(os_),
            )
        )
    return bins


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """
    ECE: count-weighted average of |confidence - accuracy| across bins. 0 = perfectly
    calibrated. This is the headline calibration number.
    """
    conf, _ = _validate(confidences, correct)
    bins = reliability_bins(conf, correct, n_bins=n_bins)
    n = len(conf)
    return sum(b.count / n * abs(b.gap) for b in bins)


def maximum_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """MCE: the worst (largest) calibration gap across all non-empty bins."""
    bins = reliability_bins(confidences, correct, n_bins=n_bins)
    return max((abs(b.gap) for b in bins), default=0.0)


def confidence_auroc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """
    Area under the ROC curve using confidence as a score for discriminating correct
    from incorrect answers. This measures *ranking* / discrimination (separate from
    calibration): can the model tell its right answers from its wrong ones?

    Computed via the Mann-Whitney U statistic with proper handling of ties, so it
    needs no external dependencies. Returns 0.5 when all answers share one label.
    """
    conf, corr = _validate(confidences, correct)
    pos = [c for c, o in zip(conf, corr) if o == 1]
    neg = [c for c, o in zip(conf, corr) if o == 0]
    if not pos or not neg:
        return 0.5  # undefined; one class is empty -> no discrimination signal

    # Rank all scores (average ranks for ties), then use the rank-sum identity.
    ordered = sorted((c, idx) for idx, c in enumerate(conf))
    ranks = [0.0] * len(conf)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[ordered[k][1]] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, o in zip(ranks, corr) if o == 1)
    n_pos, n_neg = len(pos), len(neg)
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_pos / (n_pos * n_neg)


def calibration_report(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> CalibrationReport:
    """Compute the full calibration summary in one pass-friendly call."""
    conf, corr = _validate(confidences, correct)
    acc = sum(corr) / len(corr)
    mconf = sum(conf) / len(conf)
    bins = reliability_bins(conf, corr, n_bins=n_bins)
    return CalibrationReport(
        n=len(conf),
        accuracy=acc,
        mean_confidence=mconf,
        brier=brier_score(conf, corr),
        ece=expected_calibration_error(conf, corr, n_bins=n_bins),
        mce=maximum_calibration_error(conf, corr, n_bins=n_bins),
        overconfidence=mconf - acc,
        bins=bins,
    )


def ascii_reliability_diagram(report: CalibrationReport, width: int = 40) -> str:
    """
    Render a reliability diagram as text (no matplotlib needed). Each bin shows its
    confidence range, a bar whose length is the empirical accuracy, the gap, and count.
    """
    lines = []
    lines.append("Reliability diagram (accuracy bar; '|' marks avg confidence)")
    lines.append("-" * (width + 34))
    for b in report.bins:
        filled = int(round(b.accuracy * width))
        bar = list("#" * filled + "." * (width - filled))
        conf_pos = min(width - 1, int(round(b.avg_confidence * width)))
        # Mark where average confidence falls so over/under-confidence is visible.
        bar[conf_pos] = "|" if bar[conf_pos] == "." else "I"
        sign = "+" if b.gap >= 0 else "-"
        lines.append(
            f"[{b.lo:.1f}-{b.hi:.1f}] {''.join(bar)} "
            f"acc={b.accuracy:.2f} conf={b.avg_confidence:.2f} gap={sign}{abs(b.gap):.2f} n={b.count}"
        )
    lines.append("-" * (width + 34))
    lines.append(
        f"N={report.n}  acc={report.accuracy:.3f}  conf={report.mean_confidence:.3f}  "
        f"ECE={report.ece:.3f}  MCE={report.mce:.3f}  Brier={report.brier:.3f}  "
        f"overconf={report.overconfidence:+.3f}"
    )
    return "\n".join(lines)
