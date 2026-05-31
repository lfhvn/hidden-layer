#!/usr/bin/env python3
"""
Real-model calibration: does a model know what it knows?

Elicits an answer + verbalized confidence from a real model on the bundled factual-QA
set, scores correctness offline, and reports calibration (ECE/MCE/Brier), discrimination
(AUROC), and a reliability diagram. The run is logged via the harness ExperimentTracker.

This is the production counterpart to ``demo.py`` (which validates the *metrics* with
simulated subjects). Here the subject is a real LLM via the harness provider abstraction.

Examples
--------
    # Anthropic (needs ANTHROPIC_API_KEY)
    python theory-of-mind/metacognition/run_real.py --provider anthropic --model claude-3-5-sonnet-20241022

    # Local Ollama
    python theory-of-mind/metacognition/run_real.py --provider ollama --model llama3.2:latest

    # Offline plumbing check (deterministic, no key) — numbers are not meaningful
    python theory-of-mind/metacognition/run_real.py --provider sim
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the lab importable when run directly from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from theory_of_mind.metacognition import (  # noqa: E402
    ModelSubject,
    ascii_reliability_diagram,
    confidence_auroc,
    load_dataset,
    plot_reliability,
    run_calibration,
    save_json,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run calibration on a real model.")
    parser.add_argument("--provider", default="sim", help="anthropic | openai | ollama | mlx | sim")
    parser.add_argument("--model", default=None, help="Model id (provider default if omitted)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--default-confidence", type=float, default=0.5, help="Used when the model omits a confidence")
    parser.add_argument("--no-track", action="store_true", help="Do not write a tracked run dir")
    args = parser.parse_args(argv)

    # Preflight: fail fast with a clear hint if an API key is required but missing.
    needed = _KEY_ENV.get(args.provider)
    if needed and not os.getenv(needed):
        print(
            f"Provider '{args.provider}' needs {needed}. Set it (e.g. in .env) or use "
            f"--provider sim (offline) / --provider ollama (local)."
        )
        return 1

    dataset = load_dataset()
    print(
        f"Calibration: provider={args.provider} model={args.model or '(default)'} "
        f"on {len(dataset)} questions ({dataset.name})\n"
    )

    subject = ModelSubject(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        default_confidence=args.default_confidence,
    )
    run = run_calibration(
        subject,
        dataset,
        n_bins=args.n_bins,
        track=not args.no_track,
        experiment_name="real_calibration",
        provider_label=args.provider,
        model_label=args.model or args.provider,
    )

    report = run.report
    confs = [r.confidence for r in run.records if r.confidence is not None]
    corr = [r.correct for r in run.records if r.confidence is not None]
    auroc = confidence_auroc(confs, corr) if confs else float("nan")

    print(ascii_reliability_diagram(report))
    print(
        f"\naccuracy={report.accuracy:.3f}  mean_confidence={report.mean_confidence:.3f}  "
        f"ECE={report.ece:.3f}  MCE={report.mce:.3f}  Brier={report.brier:.3f}  "
        f"AUROC={auroc:.3f}  overconfidence={report.overconfidence:+.3f}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.provider}_{(args.model or 'default').replace('/', '-').replace(':', '-')}"
    out = save_json(
        {"provider": args.provider, "model": args.model, "auroc": auroc, **run.to_dict()},
        str(OUTPUT_DIR / f"real_calibration_{tag}.json"),
    )
    print(f"\nWrote {out}")
    fig = plot_reliability(
        report, str(OUTPUT_DIR / f"real_calibration_{tag}.png"), title=f"{args.provider} {args.model or ''}".strip()
    )
    if fig:
        print(f"Wrote {fig}")
    if run.records:
        print(
            f"\nSample: Q='{run.records[0].question}' -> answer='{run.records[0].answer}' "
            f"conf={run.records[0].confidence} correct={run.records[0].correct}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
