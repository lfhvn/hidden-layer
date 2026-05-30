"""
Reporting helpers: save calibration runs to JSON and (optionally) plot reliability
diagrams. matplotlib is imported lazily and is entirely optional - if it is not
installed, ``plot_reliability`` returns None and callers fall back to the ASCII
diagram from ``metrics.ascii_reliability_diagram``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .metrics import CalibrationReport


def save_json(data: dict, path: str) -> Path:
    """Write a dict (e.g. CalibrationRun.to_dict()) to JSON, creating parent dirs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out


def plot_reliability(report: CalibrationReport, path: str, title: str = "Reliability Diagram") -> Optional[Path]:
    """
    Render a reliability diagram to an image file. Returns the path, or None if
    matplotlib is unavailable (so callers can degrade gracefully to ASCII output).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless / no display required
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")

    xs = [b.avg_confidence for b in report.bins]
    ys = [b.accuracy for b in report.bins]
    sizes = [20 + 6 * b.count for b in report.bins]
    ax.scatter(xs, ys, s=sizes, color="tab:blue", alpha=0.7, label="bins (size = count)")
    for b in report.bins:
        ax.plot([b.avg_confidence, b.avg_confidence], [b.avg_confidence, b.accuracy], color="tab:red", alpha=0.4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE={report.ece:.3f}  Brier={report.brier:.3f}  overconf={report.overconfidence:+.3f}")
    ax.legend(loc="upper left", fontsize=8)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
