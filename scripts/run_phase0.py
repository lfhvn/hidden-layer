#!/usr/bin/env python3
"""Phase-0 smoke experiment: single vs debate on 20 arithmetic tasks.

Proves the run -> track -> commit loop end to end (ROADMAP Phase 0).
Writes runs under results/multi-agent/phase0-smoke/ using the harness
tracker, so a successful run produces a committable results directory.

Usage:
    python scripts/run_phase0.py --provider ollama --model llama3.2:latest
    python scripts/run_phase0.py --strategies single --limit 5   # quick check
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from communication.multi_agent import run_strategy  # noqa: E402
from harness import ExperimentConfig, ExperimentResult, ExperimentTracker, evaluate_task  # noqa: E402

TASKS_FILE = ROOT / "shared" / "datasets" / "phase0_smoke.jsonl"
RESULTS_DIR = ROOT / "results" / "multi-agent" / "phase0-smoke"


def git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_tasks(limit: int) -> list[dict]:
    tasks = [json.loads(line) for line in TASKS_FILE.read_text().splitlines() if line.strip()]
    return tasks[:limit] if limit else tasks


def run_condition(strategy: str, tasks: list[dict], args: argparse.Namespace) -> dict:
    tracker = ExperimentTracker(base_dir=str(RESULTS_DIR))
    config = ExperimentConfig(
        experiment_name=f"phase0_{strategy}",
        task_type="arithmetic",
        strategy=strategy,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        git_hash=git_short_sha(),
        notes=f"Phase-0 smoke run; tasks={len(tasks)}; seed_note=strategies are not seeded, "
        f"repeat runs capture variance instead",
    )
    tracker.start_experiment(config)

    for task in tasks:
        start = time.time()
        try:
            result = run_strategy(strategy, task["input"], provider=args.provider, model=args.model)
            output = result.output
            error = None
            success = True
        except Exception as exc:  # keep the sweep going; record the failure
            output = ""
            error = f"{type(exc).__name__}: {exc}"
            success = False
        latency = time.time() - start

        scores = evaluate_task(task, output, eval_type=task["eval_type"]) if success else {}
        tracker.log_result(
            ExperimentResult(
                config=config,
                task_input=task["input"],
                output=output,
                latency_s=latency,
                eval_scores=scores,
                eval_metadata={"task_id": task["id"], "expected": task["expected"]},
                success=success,
                error=error,
            )
        )
        marker = "✓" if success and scores.get("accuracy") == 1.0 else "✗"
        print(f"  {marker} {task['id']} ({latency:.1f}s)")

    return tracker.finish_experiment() or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument("--strategies", default="single,debate", help="comma-separated")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    args = parser.parse_args()

    tasks = load_tasks(args.limit)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    print(f"Phase-0 smoke: {len(tasks)} tasks x {strategies} on {args.provider}/{args.model}")

    summaries = {}
    for strategy in strategies:
        print(f"\n=== {strategy} ===")
        summaries[strategy] = run_condition(strategy, tasks, args)

    print("\n=== Comparison ===")
    for strategy, summary in summaries.items():
        acc = summary.get("eval_scores", {}).get("accuracy", {}).get("mean")
        acc_str = f"{acc:.2f}" if acc is not None else "n/a"
        print(
            f"  {strategy:16s} accuracy={acc_str}  "
            f"avg_latency={summary.get('avg_latency_s', 0):.1f}s  "
            f"cost=${summary.get('total_cost_usd', 0):.4f}"
        )
    print(f"\nRuns written under {RESULTS_DIR.relative_to(ROOT)}/ — commit the run directories to make them citable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
