"""Summarize results/*.jsonl: pass rate with 95% CI, cost, and paired
comparison against the S0 baseline on the same problems.

    python -m orchestra.report
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load() -> dict[str, dict[str, dict]]:
    out = {}
    for f in sorted(RESULTS.glob("[SP]*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        out[f.stem] = {r["problem"]: r for r in rows}
    return out


def main():
    data = load()
    if not data:
        print("No results yet.")
        return
    base = data.get("S0", {})
    print(
        f"{'strat':5s} {'n':>3s} {'pass':>6s} {'95% CI':>13s} {'easy':>6s} {'med':>6s} {'hard':>6s} "
        f"{'tok/prob':>9s} {'capped':>7s} {'forced':>7s} {'calls':>6s} {'min/prob':>9s} {'vs S0 (+/-)':>12s}"
    )
    for name, rows in data.items():
        vals = list(rows.values())
        n = len(vals)
        k = sum(r["passed"] for r in vals)
        lo, hi = wilson(k, n)

        def rate(diff):
            sub = [r for r in vals if r["difficulty"] == diff]
            return f"{sum(r['passed'] for r in sub)}/{len(sub)}" if sub else "-"

        tok = sum(r["completion_tokens"] for r in vals) / n
        calls = sum(r["calls"] for r in vals) / n
        capped = sum(r["first_fail"] == "no code" for r in vals)
        forced = sum(r.get("forced", 0) > 0 for r in vals)
        mins = sum(r["model_seconds"] for r in vals) / n / 60
        shared = [pid for pid in rows if pid in base]
        gain = sum(rows[p]["passed"] and not base[p]["passed"] for p in shared)
        loss = sum(base[p]["passed"] and not rows[p]["passed"] for p in shared)
        print(
            f"{name:5s} {n:3d} {k / n:6.0%} {f'[{lo:.0%},{hi:.0%}]':>13s} {rate('easy'):>6s} "
            f"{rate('medium'):>6s} {rate('hard'):>6s} {tok:9.0f} {capped:7d} {forced:7d} {calls:6.1f} {mins:9.1f} {f'+{gain}/-{loss}':>12s}"
        )
    print("\ntok/prob = completion tokens per problem (including reasoning);")
    print("capped = problems that ended with no code; forced = problems where a call's")
    print("thinking hit the 16k budget and its answer was forced;")
    print("min/prob = model time per problem, summed over its calls.")
    print("P<n> = variable-loop research checkpoint run at n loop passes (strategy V: high effort, 16k).")


if __name__ == "__main__":
    main()
