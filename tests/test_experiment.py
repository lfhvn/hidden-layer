"""Tests for the config-driven experiment runner (harness.experiment).

All offline via the deterministic `sim` provider — no API keys, no network.
"""

import json
from pathlib import Path

import pytest

from harness import ExperimentSpec, run_experiment
from harness.experiment import Arm, load_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _qa_spec(**overrides):
    data = {
        "name": "test_qa",
        "eval_type": "keyword",
        "hypothesis": "oracle beats wrong",
        "dataset": {
            "tasks": [
                {"input": "Capital of France?", "expected": "paris"},
                {"input": "Symbol for gold?", "expected": "au"},
            ]
        },
        "arms": [
            {"name": "oracle", "provider": "sim", "params": {"sim_response": "paris au"}},
            {"name": "wrong", "provider": "sim", "params": {"sim_response": "no idea"}},
        ],
    }
    data.update(overrides)
    return ExperimentSpec.from_dict(data)


def test_run_experiment_scores_arms(tmp_path):
    report = run_experiment(_qa_spec(), base_dir=str(tmp_path), track=True)
    by_arm = {a.name: a for a in report.arms}
    assert by_arm["oracle"].scores["accuracy"] == pytest.approx(1.0)
    assert by_arm["wrong"].scores["accuracy"] == pytest.approx(0.0)
    assert by_arm["oracle"].n == 2
    assert report.hypothesis == "oracle beats wrong"


def test_run_experiment_writes_tracked_runs(tmp_path):
    report = run_experiment(_qa_spec(), base_dir=str(tmp_path), track=True)
    for arm in report.arms:
        d = Path(arm.run_dir)
        assert (d / "config.json").exists()
        assert (d / "results.jsonl").exists()
        summary = json.loads((d / "summary.json").read_text())
        # eval_scores must be numeric-only so the tracker's averaging never crashes.
        for stats in summary["eval_scores"].values():
            assert isinstance(stats["mean"], (int, float))
        # one result line per task
        lines = [ln for ln in (d / "results.jsonl").read_text().splitlines() if ln.strip()]
        assert len(lines) == 2


def test_no_track_still_aggregates(tmp_path):
    report = run_experiment(_qa_spec(), base_dir=str(tmp_path), track=False)
    assert all(a.run_dir is None for a in report.arms)
    assert report.arms[0].scores["accuracy"] == pytest.approx(1.0)
    # Nothing should have been written to disk.
    assert not any(tmp_path.iterdir())


def test_per_task_eval_override(tmp_path):
    spec = ExperimentSpec.from_dict(
        {
            "name": "mixed",
            "eval_type": "keyword",
            "dataset": {
                "tasks": [
                    {"input": "2+2?", "expected": "4", "eval_type": "numeric"},
                    {"input": "Capital of Japan?", "expected": "tokyo"},
                ]
            },
            "arms": [{"name": "a", "provider": "sim", "params": {"sim_response": "4 tokyo"}}],
        }
    )
    report = run_experiment(spec, base_dir=str(tmp_path), track=False)
    assert report.arms[0].scores["accuracy"] == pytest.approx(1.0)


def test_strategy_arm_runs_offline(tmp_path):
    spec = ExperimentSpec.from_dict(
        {
            "name": "debate_exp",
            "eval_type": "keyword",
            "dataset": {"tasks": [{"input": "Capital of France?", "expected": "paris"}]},
            "arms": [
                {
                    "name": "debate",
                    "provider": "sim",
                    "strategy": "debate",
                    "params": {"n_debaters": 2, "n_rounds": 1, "sim_response": "paris"},
                }
            ],
        }
    )
    report = run_experiment(spec, base_dir=str(tmp_path), track=False)
    assert report.arms[0].strategy == "debate"
    assert report.arms[0].n == 1


def test_failed_arm_is_recorded_not_raised(tmp_path):
    # An unknown strategy makes run_strategy raise; the runner records the failure
    # and keeps going rather than crashing the whole experiment.
    spec = ExperimentSpec.from_dict(
        {
            "name": "boom",
            "dataset": {"tasks": [{"input": "x", "expected": "y"}]},
            "arms": [{"name": "bad", "provider": "sim", "strategy": "does_not_exist"}],
        }
    )
    report = run_experiment(spec, base_dir=str(tmp_path), track=True)
    assert report.arms[0].n == 0  # no successful scored tasks
    summary = json.loads((Path(report.arms[0].run_dir) / "summary.json").read_text())
    assert summary["success_rate"] == 0.0


def test_arm_call_kwargs_precedence():
    arm = Arm(name="x", provider="sim", model="m1", params={"model": "m2", "temperature": 0.1})
    kwargs = arm.call_kwargs()
    assert kwargs["provider"] == "sim"
    assert kwargs["model"] == "m2"  # explicit params override
    assert kwargs["temperature"] == 0.1


def test_load_spec_from_example_file():
    spec = load_spec(str(REPO_ROOT / "config" / "experiments" / "example_qa.yaml"))
    assert spec.name == "example_qa"
    assert len(spec.arms) == 3
    assert {a.name for a in spec.arms} == {"oracle", "wrong", "debate"}


def test_empty_spec_rejected():
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict({"name": "empty", "dataset": {"tasks": []}, "arms": []})
