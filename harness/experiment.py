"""
Config-driven experiment runner.

This is the "test" half of the hypothesis -> test -> publish loop. You describe an
experiment once (a hypothesis, a dataset of tasks, an eval, and one or more *arms* to
compare) and ``run_experiment`` executes the full matrix, scores every output with the
harness eval functions, and logs each arm as a reproducible ``ExperimentTracker`` run
(``config.json`` / ``results.jsonl`` / ``summary.json``). The returned ``ExperimentReport``
also carries in-memory aggregates so callers/tests don't have to re-read disk.

Pair it with ``harness.analysis`` to turn the logged runs into paper-ready tables.

Spec schema (YAML or dict)::

    name: factual_qa_v1
    hypothesis: "Lower temperature improves exact-match accuracy on factual QA."
    task_type: qa
    eval_type: keyword            # default eval for tasks that don't set their own
    dataset:
      tasks:                      # inline, or use `path: tasks.jsonl`
        - input: "What is the capital of France?"
          expected: paris
        - input: "What is 2 + 2?"
          expected: "4"
          eval_type: numeric      # per-task override
    arms:                         # each arm becomes one tracked run
      - name: sim-correct
        provider: sim
        params: {sim_response: "paris"}     # extra llm_call kwargs pass straight through
      - name: debate
        provider: sim
        strategy: debate                    # if set, route through run_strategy()
        params: {n_debaters: 2, n_rounds: 1, sim_response: "paris"}
      - name: sonnet
        config: claude-sonnet               # reference a config/models.yaml preset

Run from the CLI::

    python -m harness.experiment run path/to/spec.yaml
    python -m harness.experiment report "experiments/factual_qa_v1.*" --format markdown
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evals import evaluate_task
from .experiment_tracker import ExperimentConfig, ExperimentResult, ExperimentTracker
from .llm_provider import llm_call
from .model_config import get_model_config

# --------------------------------------------------------------------------- spec


@dataclass
class Arm:
    """One condition to run over the whole dataset (becomes one tracked run)."""

    name: str
    provider: Optional[str] = None
    model: Optional[str] = None
    config: Optional[str] = None  # name of a config/models.yaml preset
    strategy: Optional[str] = None  # if set, call run_strategy() instead of llm_call()
    params: Dict[str, Any] = field(default_factory=dict)  # extra kwargs passed through

    def call_kwargs(self) -> Dict[str, Any]:
        """Resolve the kwargs handed to llm_call/run_strategy for this arm."""
        kwargs: Dict[str, Any] = {}
        if self.config:
            kwargs.update(get_model_config(self.config).to_kwargs())
        if self.provider:
            kwargs["provider"] = self.provider
        if self.model:
            kwargs["model"] = self.model
        kwargs.update(self.params)  # explicit params win
        return kwargs


@dataclass
class ExperimentSpec:
    """A complete, reproducible experiment description."""

    name: str
    tasks: List[Dict[str, Any]]
    arms: List[Arm]
    eval_type: str = "exact_match"
    task_type: str = "general"
    hypothesis: str = ""
    notes: str = ""

    @staticmethod
    def from_dict(data: Dict[str, Any], base_dir: Optional[Path] = None) -> "ExperimentSpec":
        tasks = _load_tasks(data.get("dataset", {}), base_dir)
        if not tasks:
            raise ValueError("Experiment spec has no tasks (set dataset.tasks or dataset.path).")
        arms_raw = data.get("arms") or []
        if not arms_raw:
            raise ValueError("Experiment spec has no arms.")
        arms = []
        for i, a in enumerate(arms_raw):
            arms.append(
                Arm(
                    name=str(a.get("name") or _default_arm_name(a, i)),
                    provider=a.get("provider"),
                    model=a.get("model"),
                    config=a.get("config"),
                    strategy=a.get("strategy"),
                    params=dict(a.get("params") or {}),
                )
            )
        return ExperimentSpec(
            name=data["name"],
            tasks=tasks,
            arms=arms,
            eval_type=data.get("eval_type", "exact_match"),
            task_type=data.get("task_type", "general"),
            hypothesis=data.get("hypothesis", ""),
            notes=data.get("notes", ""),
        )


def load_spec(path: str) -> ExperimentSpec:
    """Load an experiment spec from a YAML (or JSON) file."""
    import yaml

    p = Path(path)
    data = yaml.safe_load(p.read_text())
    return ExperimentSpec.from_dict(data, base_dir=p.parent)


def _default_arm_name(a: Dict[str, Any], i: int) -> str:
    parts = [a.get("config") or a.get("model") or a.get("provider") or f"arm{i}"]
    if a.get("strategy"):
        parts.append(str(a["strategy"]))
    return "-".join(parts)


def _load_tasks(dataset: Dict[str, Any], base_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if dataset.get("tasks"):
        return [dict(t) for t in dataset["tasks"]]
    path = dataset.get("path")
    if not path:
        return []
    fp = Path(path)
    if base_dir and not fp.is_absolute():
        fp = base_dir / fp
    text = fp.read_text()
    if fp.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data["tasks"] if isinstance(data, dict) else data


# --------------------------------------------------------------------------- report


@dataclass
class ArmReport:
    """Aggregated outcome for a single arm."""

    name: str
    provider: str
    model: str
    strategy: str
    n: int
    scores: Dict[str, float]  # metric -> mean
    avg_latency_s: float
    total_cost_usd: float
    run_dir: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "arm": self.name,
            "provider": self.provider,
            "model": self.model,
            "strategy": self.strategy,
            "n": self.n,
        }
        row.update({k: round(v, 4) for k, v in self.scores.items()})
        row["latency_s"] = round(self.avg_latency_s, 4)
        row["cost_usd"] = round(self.total_cost_usd, 6)
        return row


@dataclass
class ExperimentReport:
    """Result of running every arm over the dataset."""

    name: str
    hypothesis: str
    arms: List[ArmReport]

    def to_rows(self) -> List[Dict[str, Any]]:
        return [a.to_row() for a in self.arms]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "arms": [a.to_row() for a in self.arms],
            "run_dirs": {a.name: a.run_dir for a in self.arms},
        }


# --------------------------------------------------------------------------- run


def _git_hash() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git absent / not a repo
        return None


def _means(per_task: List[Dict[str, float]]) -> Dict[str, float]:
    keys: set = set()
    for d in per_task:
        keys.update(d.keys())
    out: Dict[str, float] = {}
    for k in sorted(keys):
        vals = [d[k] for d in per_task if k in d]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def _call_arm(arm: Arm, task_input: str, kwargs: Dict[str, Any]):
    """Run a single (arm, task). Returns (output, tokens_in, tokens_out, cost, latency)."""
    start = time.time()
    if arm.strategy:
        from harness import run_strategy  # lazy; keeps harness import-light

        res = run_strategy(arm.strategy, task_input=task_input, **kwargs)
        output = res.output
        tokens_in = getattr(res, "tokens_in", None)
        tokens_out = getattr(res, "tokens_out", None)
        cost = getattr(res, "cost_usd", None)
        latency = time.time() - start
    else:
        res = llm_call(task_input, **kwargs)
        output = res.text
        tokens_in, tokens_out, cost = res.tokens_in, res.tokens_out, res.cost_usd
        latency = res.latency_s or (time.time() - start)
    return output, tokens_in, tokens_out, cost, latency


def run_experiment(
    spec: ExperimentSpec,
    base_dir: str = "./experiments",
    track: bool = True,
) -> ExperimentReport:
    """
    Execute every arm over every task, score outputs, and (optionally) log tracked runs.

    Args:
        spec: the experiment to run.
        base_dir: where ExperimentTracker writes run directories.
        track: if False, skip on-disk logging (still returns aggregates).
    """
    tracker = ExperimentTracker(base_dir=base_dir) if track else None
    arm_reports: List[ArmReport] = []

    for arm in spec.arms:
        kwargs = arm.call_kwargs()
        cfg = ExperimentConfig(
            experiment_name=f"{spec.name}.{arm.name}",
            task_type=spec.task_type,
            strategy=arm.strategy or "single",
            provider=str(kwargs.get("provider", "default")),
            model=str(kwargs.get("model", "default")),
            temperature=float(kwargs.get("temperature", 0.7)),
            git_hash=_git_hash(),
            notes=spec.hypothesis,
        )
        run_dir = str(tracker.start_experiment(cfg)) if tracker else None

        per_task_scores: List[Dict[str, float]] = []
        latencies: List[float] = []
        costs: List[float] = []
        for idx, task in enumerate(spec.tasks):
            task_input = task["input"]
            try:
                output, t_in, t_out, cost, latency = _call_arm(arm, task_input, kwargs)
                scores = evaluate_task(task, output, eval_type=task.get("eval_type", spec.eval_type))
                # Keep eval_scores numeric (the tracker averages them); strings -> metadata.
                numeric = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
                meta = {k: v for k, v in scores.items() if k not in numeric}
                per_task_scores.append(numeric)
                latencies.append(latency)
                if cost:
                    costs.append(cost)
                if tracker:
                    tracker.log_result(
                        ExperimentResult(
                            config=cfg,
                            task_input=task_input,
                            output=output,
                            latency_s=latency,
                            tokens_in=t_in,
                            tokens_out=t_out,
                            cost_usd=cost,
                            eval_scores=numeric,
                            eval_metadata={"task_index": idx, **meta},
                        )
                    )
            except Exception as exc:  # record the failure, keep the matrix going
                if tracker:
                    tracker.log_result(
                        ExperimentResult(
                            config=cfg,
                            task_input=task_input,
                            output="",
                            latency_s=0.0,
                            success=False,
                            error=str(exc),
                        )
                    )

        if tracker:
            tracker.finish_experiment()

        arm_reports.append(
            ArmReport(
                name=arm.name,
                provider=cfg.provider,
                model=cfg.model,
                strategy=cfg.strategy,
                n=len(per_task_scores),
                scores=_means(per_task_scores),
                avg_latency_s=sum(latencies) / len(latencies) if latencies else 0.0,
                total_cost_usd=sum(costs),
                run_dir=run_dir,
            )
        )

    return ExperimentReport(name=spec.name, hypothesis=spec.hypothesis, arms=arm_reports)


def run_experiment_file(path: str, base_dir: str = "./experiments", track: bool = True) -> ExperimentReport:
    """Convenience: load a spec file and run it."""
    return run_experiment(load_spec(path), base_dir=base_dir, track=track)


# --------------------------------------------------------------------------- cli


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="harness.experiment", description="Config-driven experiment runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run an experiment spec (YAML/JSON)")
    p_run.add_argument("spec")
    p_run.add_argument("--out", default="./experiments", help="Experiment output dir")
    p_run.add_argument("--no-track", action="store_true", help="Do not write run dirs")
    p_run.add_argument("--format", default="markdown", choices=["markdown", "latex", "csv"])

    p_rep = sub.add_parser("report", help="Tabulate previously logged runs")
    p_rep.add_argument("glob", help="Glob of experiment run dirs, e.g. 'experiments/myexp.*'")
    p_rep.add_argument("--format", default="markdown", choices=["markdown", "latex", "csv"])

    args = parser.parse_args(argv)

    from . import analysis

    if args.cmd == "run":
        report = run_experiment_file(args.spec, base_dir=args.out, track=not args.no_track)
        if report.hypothesis:
            print(f"\nHypothesis: {report.hypothesis}\n")
        print(analysis.format_rows(report.to_rows(), args.format))
        return 0

    if args.cmd == "report":
        rows = analysis.runs_to_rows(analysis.load_runs(args.glob))
        print(analysis.format_rows(rows, args.format))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
