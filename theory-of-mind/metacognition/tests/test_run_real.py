"""Offline test for the real-model calibration entrypoint (run_real.py).

Exercises the entrypoint via the deterministic ``sim`` provider so the plumbing
(elicit -> parse -> score -> calibrate -> report) is covered without API keys.
"""

import importlib.util
from pathlib import Path

_RUN_REAL = Path(__file__).resolve().parents[1] / "run_real.py"
_spec = importlib.util.spec_from_file_location("metacog_run_real", _RUN_REAL)
run_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_real)


def test_run_real_offline_smoke():
    # sim provider -> deterministic, offline; numbers are degenerate but the pipeline runs.
    assert run_real.main(["--provider", "sim", "--no-track"]) == 0


def test_run_real_requires_key_for_api_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Preflight should refuse (exit 1) rather than attempt a keyless API call.
    assert run_real.main(["--provider", "anthropic"]) == 1
