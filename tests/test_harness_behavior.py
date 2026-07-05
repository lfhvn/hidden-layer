"""Behavioral tests for harness modules with no prior coverage.

These exercise real logic (file IO, aggregation) without any LLM calls:
- system_prompts: the prompt directory must resolve inside the repo and
  named prompts must load (a path bug once disabled this silently).
- experiment_tracker: full start -> log -> finish cycle, including
  aggregation over metadata and non-numeric score values.
- evals.evaluate_task: dispatch for the non-LLM eval types.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestSystemPrompts:
    """System prompt loading (config/system_prompts)."""

    def test_prompts_dir_is_inside_repo(self):
        from harness.system_prompts import SYSTEM_PROMPTS_DIR

        assert SYSTEM_PROMPTS_DIR.exists(), f"missing: {SYSTEM_PROMPTS_DIR}"
        assert ROOT in SYSTEM_PROMPTS_DIR.parents

    def test_list_contains_bundled_prompts(self):
        from harness.system_prompts import list_system_prompts

        names = list_system_prompts()
        assert "default" in names
        assert "researcher" in names

    def test_load_named_prompt_returns_content(self):
        from harness.system_prompts import load_system_prompt

        content = load_system_prompt("researcher")
        assert len(content) > 50
        assert content != "researcher"

    def test_resolve_passes_through_raw_prompts(self):
        from harness.system_prompts import resolve_system_prompt

        raw = "You are a helpful assistant. Answer concisely."
        assert resolve_system_prompt(raw) == raw
        assert resolve_system_prompt(None) is None

    def test_resolve_expands_named_prompt(self):
        from harness.system_prompts import load_system_prompt, resolve_system_prompt

        assert resolve_system_prompt("researcher") == load_system_prompt("researcher")


class TestExperimentTracker:
    """Full tracking cycle against a temporary directory."""

    def _make_result(self, config, output, accuracy, extra_scores=None):
        from harness.experiment_tracker import ExperimentResult

        scores = {"accuracy": accuracy}
        if extra_scores:
            scores.update(extra_scores)
        return ExperimentResult(
            config=config,
            task_input="2 + 2?",
            output=output,
            latency_s=0.5,
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.001,
            eval_scores=scores,
        )

    def _make_config(self):
        from harness.experiment_tracker import ExperimentConfig

        return ExperimentConfig(
            experiment_name="tracker_behavior_test",
            task_type="reasoning",
            strategy="single",
            provider="ollama",
            model="test-model",
        )

    def test_full_cycle_writes_config_results_summary(self, tmp_path):
        from harness.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(base_dir=str(tmp_path))
        config = self._make_config()
        run_dir = tracker.start_experiment(config)

        for acc in (1.0, 0.0, 1.0):
            tracker.log_result(self._make_result(config, "4", acc))
        summary = tracker.finish_experiment()

        assert (run_dir / "config.json").exists()
        assert (run_dir / "summary.json").exists()
        lines = (run_dir / "results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["eval_scores"]["accuracy"] == 1.0

        agg = summary["eval_scores"]["accuracy"]
        assert agg["count"] == 3
        assert abs(agg["mean"] - 2 / 3) < 1e-9
        assert agg["min"] == 0.0
        assert agg["max"] == 1.0

    def test_non_numeric_metadata_scores_do_not_crash_summary(self, tmp_path):
        """Underscore-prefixed / string-valued scores (e.g. _judge_reasoning)
        must be excluded from aggregation instead of crashing min()/sum()."""
        from harness.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(base_dir=str(tmp_path))
        config = self._make_config()
        tracker.start_experiment(config)
        tracker.log_result(
            self._make_result(config, "4", 1.0, extra_scores={"_judge_reasoning": "sound arithmetic"})
        )
        tracker.log_result(self._make_result(config, "5", 0.0))
        summary = tracker.finish_experiment()

        assert "_judge_reasoning" not in summary["eval_scores"]
        assert summary["eval_scores"]["accuracy"]["count"] == 2

    def test_scores_missing_from_first_result_still_aggregate(self, tmp_path):
        from harness.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(base_dir=str(tmp_path))
        config = self._make_config()
        tracker.start_experiment(config)
        first = self._make_result(config, "4", 1.0)
        first.eval_scores = {}
        tracker.log_result(first)
        tracker.log_result(self._make_result(config, "4", 1.0))
        summary = tracker.finish_experiment()

        assert summary["eval_scores"]["accuracy"]["count"] == 1

    def test_finish_without_results_is_a_noop(self, tmp_path):
        from harness.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(base_dir=str(tmp_path))
        tracker.start_experiment(self._make_config())
        assert tracker.finish_experiment() is None


class TestEvaluateTaskDispatch:
    """evaluate_task routing for eval types that need no LLM."""

    def test_exact_match_task(self):
        from harness import evaluate_task

        scores = evaluate_task({"input": "q", "expected": "Paris", "eval_type": "exact_match"}, "paris")
        assert scores["accuracy"] == 1.0
        assert "coherence" in scores

    def test_keyword_task_accepts_str_or_list(self):
        from harness import evaluate_task

        task = {"input": "q", "expected": "blue", "eval_type": "keyword"}
        assert evaluate_task(task, "the sky is blue")["accuracy"] == 1.0
        task_list = {"input": "q", "expected": ["blue", "red"], "eval_type": "keyword"}
        assert evaluate_task(task_list, "the sky is blue")["accuracy"] == 1.0

    def test_numeric_task(self):
        from harness import evaluate_task

        task = {"input": "q", "expected": "42", "eval_type": "numeric"}
        assert evaluate_task(task, "The answer is 42.")["accuracy"] == 1.0
        assert evaluate_task(task, "The answer is 41.")["accuracy"] == 0.0

    def test_all_numeric_scores_are_floats(self):
        from harness import evaluate_task

        scores = evaluate_task({"input": "q", "expected": "x", "eval_type": "exact_match"}, "x")
        for key, value in scores.items():
            if not key.startswith("_"):
                assert isinstance(value, (int, float)), f"{key} is {type(value)}"
