#!/usr/bin/env python3
"""
Offline metacognition demo - runs with zero API keys and zero network.

It uses the harness ``sim`` provider via deterministic SimulatedSubjects to show:
  1. Calibration metrics (ECE/MCE/Brier/AUROC) across four calibration profiles.
  2. An ASCII reliability diagram (and a matplotlib PNG if matplotlib is installed).
  3. A self-prediction (introspection) experiment across self-knowledge levels.

Run:  python theory-of-mind/metacognition/demo.py
Output (JSON + optional PNG) is written to theory-of-mind/metacognition/output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the lab importable when run directly from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from theory_of_mind.metacognition import (  # noqa: E402
    SimulatedIntrospector,
    SimulatedSubject,
    ascii_reliability_diagram,
    confidence_auroc,
    make_arithmetic_dataset,
    plot_reliability,
    run_calibration,
    run_self_prediction,
    save_json,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROFILES = ["calibrated", "overconfident", "underconfident", "random"]


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    dataset = make_arithmetic_dataset(n=1500, seed=42)

    section("1. CALIBRATION ACROSS PROFILES")
    print(f"Dataset: {dataset.name} ({len(dataset)} deterministic arithmetic questions)")
    print(f"{'profile':16s}{'acc':>7s}{'conf':>7s}{'ECE':>7s}{'MCE':>7s}{'Brier':>7s}{'AUROC':>7s}")
    print("-" * 57)

    runs = {}
    for profile in PROFILES:
        subject = SimulatedSubject(accuracy=0.7, profile=profile, seed=7)
        run = run_calibration(subject, dataset, n_bins=10)
        runs[profile] = run
        r = run.report
        confs = [rec.confidence for rec in run.records]
        corr = [rec.correct for rec in run.records]
        auroc = confidence_auroc(confs, corr)
        print(
            f"{profile:16s}{r.accuracy:7.3f}{r.mean_confidence:7.3f}"
            f"{r.ece:7.3f}{r.mce:7.3f}{r.brier:7.3f}{auroc:7.3f}"
        )

    print("\nInterpretation:")
    print("  - 'calibrated' should have ECE near 0 and confidence ~ accuracy.")
    print("  - 'overconfident' inflates confidence above accuracy (positive gaps).")
    print("  - 'underconfident' deflates confidence below accuracy.")
    print("  - 'random' confidences carry no signal -> AUROC ~ 0.5.")

    section("2. RELIABILITY DIAGRAM (overconfident subject)")
    over = runs["overconfident"].report
    print(ascii_reliability_diagram(over))

    section("3. SELF-PREDICTION (INTROSPECTION)")
    print("Can a subject predict its own answer before giving it?")
    print(f"{'self_knowledge':>16s}{'self_pred_acc':>16s}{'task_acc':>12s}")
    print("-" * 44)
    sp_runs = {}
    for sk in [0.95, 0.7, 0.5, 0.2]:
        intro = SimulatedIntrospector(accuracy=0.7, self_knowledge=sk, seed=7)
        spr = run_self_prediction(intro, dataset)
        sp_runs[sk] = spr
        print(f"{sk:16.2f}{spr.self_prediction_accuracy:16.3f}{spr.task_accuracy:12.3f}")
    print("\nNote: self-prediction accuracy tracks self_knowledge and is")
    print("independent of task accuracy - introspection is not just being right.")

    # Persist artifacts.
    section("4. ARTIFACTS")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "calibration": {p: runs[p].to_dict()["report"] for p in PROFILES},
        "self_prediction": {
            str(sk): {
                "self_prediction_accuracy": sp_runs[sk].self_prediction_accuracy,
                "task_accuracy": sp_runs[sk].task_accuracy,
            }
            for sk in sp_runs
        },
    }
    json_path = save_json(summary, str(OUTPUT_DIR / "demo_summary.json"))
    print(f"Wrote summary JSON: {json_path}")

    png_path = plot_reliability(over, str(OUTPUT_DIR / "reliability_overconfident.png"), title="Overconfident subject")
    if png_path:
        print(f"Wrote reliability diagram: {png_path}")
    else:
        print("matplotlib not installed - skipped PNG (ASCII diagram shown above).")

    print("\nDone. Everything above ran offline via the harness 'sim' provider.")


if __name__ == "__main__":
    main()
