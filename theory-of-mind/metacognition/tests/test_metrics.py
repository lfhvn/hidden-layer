"""Tests for calibration metrics - mostly closed-form, known-answer cases."""

import pytest

from theory_of_mind.metacognition import (
    ascii_reliability_diagram,
    brier_score,
    calibration_report,
    confidence_auroc,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)


def test_brier_perfect_and_worst():
    # Perfect: confidence 1 on correct, 0 on incorrect.
    assert brier_score([1.0, 0.0], [True, False]) == 0.0
    # Worst: confidence 1 on incorrect, 0 on correct -> squared error 1 each.
    assert brier_score([1.0, 0.0], [False, True]) == 1.0
    # Always 0.5 -> 0.25 regardless of outcome.
    assert brier_score([0.5, 0.5], [True, False]) == 0.25


def test_ece_zero_when_perfectly_calibrated():
    # Two bins, in each the accuracy equals the (constant) confidence.
    # bin ~0.2: 5 items, 1 correct -> accuracy 0.2, confidence 0.2
    # bin ~0.8: 5 items, 4 correct -> accuracy 0.8, confidence 0.8
    conf = [0.2] * 5 + [0.8] * 5
    corr = [True, False, False, False, False] + [True, True, True, True, False]
    assert expected_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_simple_known_value():
    # All confidence 0.9, accuracy 0.5 -> single bin gap 0.4 -> ECE 0.4.
    conf = [0.9, 0.9, 0.9, 0.9]
    corr = [True, True, False, False]
    assert expected_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.4)
    assert maximum_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.4)


def test_reliability_bins_skip_empty_and_cover_endpoints():
    conf = [0.05, 0.95, 1.0]
    corr = [False, True, True]
    bins = reliability_bins(conf, corr, n_bins=10)
    # Only the first and last bins are populated; 1.0 lands in the last bin.
    assert len(bins) == 2
    last = bins[-1]
    assert last.count == 2 and last.accuracy == pytest.approx(1.0)


def test_auroc_perfect_chance_and_reversed():
    # Confidence perfectly separates correct (high) from incorrect (low).
    assert confidence_auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == pytest.approx(1.0)
    # Reversed -> 0.0.
    assert confidence_auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == pytest.approx(0.0)
    # All ties -> 0.5.
    assert confidence_auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == pytest.approx(0.5)
    # Single class present -> undefined, defined as 0.5.
    assert confidence_auroc([0.9, 0.8], [True, True]) == 0.5


def test_overconfidence_sign():
    rep = calibration_report([0.9, 0.9, 0.9], [True, False, False])
    assert rep.overconfidence > 0  # confident but mostly wrong
    assert rep.accuracy == pytest.approx(1 / 3)
    assert rep.mean_confidence == pytest.approx(0.9)


def test_validation_errors():
    with pytest.raises(ValueError):
        brier_score([], [])
    with pytest.raises(ValueError):
        brier_score([0.5], [True, False])  # length mismatch
    with pytest.raises(ValueError):
        brier_score([1.5], [True])  # out of range


def test_ascii_diagram_is_string():
    rep = calibration_report([0.2, 0.8, 0.9], [False, True, True])
    out = ascii_reliability_diagram(rep)
    assert isinstance(out, str) and "ECE=" in out
