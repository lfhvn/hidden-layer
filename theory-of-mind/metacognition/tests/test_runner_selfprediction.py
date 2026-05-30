"""Integration tests for the calibration runner and self-prediction task."""

import pytest

from theory_of_mind.metacognition import (
    SimulatedIntrospector,
    SimulatedSubject,
    confidence_auroc,
    make_arithmetic_dataset,
    run_calibration,
    run_self_prediction,
)


def test_calibrated_subject_has_low_ece_at_large_n():
    ds = make_arithmetic_dataset(n=3000, seed=11)
    run = run_calibration(SimulatedSubject(accuracy=0.7, profile="calibrated", seed=9), ds)
    # In the large-N limit a calibrated subject is well-calibrated.
    assert run.report.ece < 0.05
    assert run.report.accuracy == pytest.approx(run.report.mean_confidence, abs=0.05)


def test_overconfident_has_higher_ece_than_calibrated():
    ds = make_arithmetic_dataset(n=2000, seed=12)
    cal = run_calibration(SimulatedSubject(0.7, "calibrated", seed=1), ds).report
    over = run_calibration(SimulatedSubject(0.7, "overconfident", seed=1), ds).report
    assert over.ece > cal.ece
    assert over.overconfidence > cal.overconfidence


def test_random_profile_has_chance_auroc():
    ds = make_arithmetic_dataset(n=2000, seed=13)
    run = run_calibration(SimulatedSubject(0.7, "random", seed=1), ds)
    confs = [r.confidence for r in run.records]
    corr = [r.correct for r in run.records]
    assert confidence_auroc(confs, corr) == pytest.approx(0.5, abs=0.05)


def test_runner_record_count_matches_dataset():
    ds = make_arithmetic_dataset(n=40, seed=14)
    run = run_calibration(SimulatedSubject(0.6, "calibrated", seed=2), ds)
    assert len(run.records) == len(ds)
    assert run.dataset_name == ds.name


def test_self_prediction_tracks_self_knowledge():
    ds = make_arithmetic_dataset(n=1000, seed=15)
    high = run_self_prediction(SimulatedIntrospector(0.7, self_knowledge=0.95, seed=4), ds)
    low = run_self_prediction(SimulatedIntrospector(0.7, self_knowledge=0.1, seed=4), ds)
    assert high.self_prediction_accuracy > 0.9
    assert low.self_prediction_accuracy < 0.2
    # Task accuracy is the same regardless of self-knowledge (seed-matched).
    assert high.task_accuracy == pytest.approx(low.task_accuracy, abs=1e-9)


def test_self_prediction_reproducible():
    ds = make_arithmetic_dataset(n=100, seed=16)
    a = run_self_prediction(SimulatedIntrospector(0.7, 0.8, seed=4), ds)
    b = run_self_prediction(SimulatedIntrospector(0.7, 0.8, seed=4), ds)
    assert a.to_dict() == b.to_dict()
