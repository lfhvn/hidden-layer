"""
Analysis & reporting for tracked experiments.

This is the "publish" half of the loop. It reads the run directories written by
``ExperimentTracker`` (``config.json`` + ``summary.json``) and turns them into
paper-ready artifacts:

  - ``load_runs(glob)`` -> ``[RunData]``           load logged runs from disk
  - ``runs_to_rows(runs)`` -> ``[dict]``           one tidy row per run/arm
  - ``format_rows(rows, "markdown"|"latex"|"csv")`` render a comparison table
  - ``latex_macros(rows, metrics)``                emit ``\\newcommand`` result macros
  - ``bar_plot(rows, metric, path)``               optional matplotlib chart (ASCII fallback)

Pure stdlib (no numpy/pandas); matplotlib is optional. ``rows`` is the lingua franca:
both ``ExperimentReport.to_rows()`` (in-memory) and ``runs_to_rows`` (from disk) produce
the same shape, so every formatter works on either source.
"""

from __future__ import annotations

import csv
import glob as _glob
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Columns that are identifiers/metadata rather than reported metrics.
_META_COLUMNS = ("arm", "provider", "model", "strategy", "n", "run", "experiment")


@dataclass
class RunData:
    """A single experiment run loaded from disk."""

    run_dir: str
    config: Dict[str, Any]
    summary: Dict[str, Any]

    @property
    def name(self) -> str:
        return self.config.get("experiment_name", Path(self.run_dir).name)


def load_run(run_dir: str) -> RunData:
    """Load one run directory (must contain config.json and summary.json)."""
    p = Path(run_dir)
    config = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else {}
    summary = json.loads((p / "summary.json").read_text()) if (p / "summary.json").exists() else {}
    return RunData(run_dir=str(p), config=config, summary=summary)


def load_runs(pattern: str) -> List[RunData]:
    """
    Load all runs matching a glob. Accepts either a directory (loads its run
    subdirectories) or a glob of run dirs (e.g. ``experiments/myexp.*``).
    Only directories containing a ``summary.json`` are included; results are sorted.
    """
    candidates: List[str] = []
    p = Path(pattern)
    if p.is_dir() and (p / "summary.json").exists():
        candidates = [str(p)]
    elif p.is_dir():
        candidates = [str(c) for c in p.iterdir() if c.is_dir()]
    else:
        candidates = _glob.glob(pattern)
    runs = [load_run(c) for c in sorted(candidates) if (Path(c) / "summary.json").exists()]
    return runs


def _arm_label(config: Dict[str, Any], run_dir: str) -> str:
    """Derive a short arm label. experiment_name is 'spec.arm' -> take the arm part."""
    name = config.get("experiment_name", Path(run_dir).name)
    return name.split(".", 1)[1] if "." in name else name


def runs_to_rows(runs: List[RunData]) -> List[Dict[str, Any]]:
    """Turn loaded runs into tidy rows (mean of each eval metric + latency/cost)."""
    rows: List[Dict[str, Any]] = []
    for r in runs:
        row: Dict[str, Any] = {
            "arm": _arm_label(r.config, r.run_dir),
            "provider": r.config.get("provider", ""),
            "model": r.config.get("model", ""),
            "strategy": r.config.get("strategy", ""),
            "n": r.summary.get("total_runs", 0),
        }
        for metric, stats in (r.summary.get("eval_scores") or {}).items():
            if isinstance(stats, dict) and "mean" in stats:
                row[metric] = round(stats["mean"], 4)
        if r.summary.get("avg_latency_s"):
            row["latency_s"] = round(r.summary["avg_latency_s"], 4)
        if r.summary.get("total_cost_usd"):
            row["cost_usd"] = round(r.summary["total_cost_usd"], 6)
        rows.append(row)
    return rows


# ------------------------------------------------------------------- formatting


def _columns(rows: List[Dict[str, Any]], columns: Optional[List[str]]) -> List[str]:
    if columns:
        return columns
    # Stable order: known meta first, then any metric columns in first-seen order.
    seen: List[str] = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.append(k)
    meta = [c for c in _META_COLUMNS if c in seen]
    metrics = [c for c in seen if c not in _META_COLUMNS]
    return meta + metrics


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return "" if v is None else str(v)


def to_markdown(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    """Render rows as a GitHub-flavored markdown table."""
    if not rows:
        return "_(no runs)_"
    cols = _columns(rows, columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def to_csv(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    """Render rows as CSV text."""
    if not rows:
        return ""
    cols = _columns(rows, columns)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: _fmt(row.get(c)) for c in cols})
    return buf.getvalue().strip("\n")


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def to_latex(
    rows: List[Dict[str, Any]], columns: Optional[List[str]] = None, caption: str = "", label: str = ""
) -> str:
    """Render rows as a LaTeX ``table`` with a ``tabular`` body."""
    if not rows:
        return "% (no runs)"
    cols = _columns(rows, columns)
    align = "l" * len([c for c in cols if c in _META_COLUMNS]) + "r" * (
        len(cols) - len([c for c in cols if c in _META_COLUMNS])
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(_fmt(row.get(c))) for c in cols) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def format_rows(rows: List[Dict[str, Any]], fmt: str = "markdown", **kwargs) -> str:
    """Dispatch to to_markdown / to_latex / to_csv by name."""
    fmt = fmt.lower()
    if fmt in ("markdown", "md"):
        return to_markdown(rows, **kwargs)
    if fmt in ("latex", "tex"):
        return to_latex(rows, **kwargs)
    if fmt == "csv":
        return to_csv(rows, **kwargs)
    raise ValueError(f"Unknown format {fmt!r}; use markdown, latex, or csv.")


# ------------------------------------------------------------- paper binding


def _camel(s: str) -> str:
    """Sanitize an identifier into a LaTeX-macro-safe CamelCase token (letters only)."""
    parts = "".join(c if c.isalnum() else " " for c in str(s)).split()
    token = "".join(p[:1].upper() + p[1:] for p in parts)
    # LaTeX command names must be letters only; map digits to words.
    digit_words = {
        "0": "Zero",
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
    }
    return "".join(digit_words.get(ch, ch) for ch in token) or "X"


def latex_macros(
    rows: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    prefix: str = "result",
) -> str:
    """
    Emit ``\\newcommand`` definitions binding each (arm, metric) to its value, so papers
    cite numbers that come *from runs* instead of being transcribed by hand.

    Example: arm "sonnet", metric "accuracy" -> ``\\newcommand{\\resultSonnetAccuracy}{0.92}``.
    """
    if metrics is None:
        metric_set: List[str] = []
        for row in rows:
            for k in row:
                if k not in _META_COLUMNS and k not in metric_set:
                    metric_set.append(k)
        metrics = metric_set
    lines: List[str] = []
    for row in rows:
        arm = row.get("arm", "arm")
        for m in metrics:
            if m in row and isinstance(row[m], (int, float)):
                macro = f"\\{prefix}{_camel(arm)}{_camel(m)}"
                lines.append(f"\\newcommand{{{macro}}}{{{_fmt(row[m])}}}")
    return "\n".join(lines)


# ------------------------------------------------------------- plotting (optional)


def _ascii_bar(rows: List[Dict[str, Any]], metric: str, width: int = 40) -> str:
    vals = [(row.get("arm", "?"), row.get(metric)) for row in rows if isinstance(row.get(metric), (int, float))]
    if not vals:
        return f"(no '{metric}' values)"
    hi = max(v for _, v in vals) or 1.0
    label_w = max(len(str(a)) for a, _ in vals)
    lines = [f"{metric}:"]
    for arm, v in vals:
        filled = int(round((v / hi) * width)) if hi else 0
        lines.append(f"  {str(arm):<{label_w}}  {'#' * filled} {_fmt(v)}")
    return "\n".join(lines)


def bar_plot(rows: List[Dict[str, Any]], metric: str, path: str, title: str = "") -> Optional[str]:
    """
    Save a bar chart comparing ``metric`` across arms. Returns the path, or None if
    matplotlib is unavailable (callers can fall back to ``_ascii_bar``).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    vals = [(str(row.get("arm", "?")), row.get(metric)) for row in rows if isinstance(row.get(metric), (int, float))]
    if not vals:
        return None
    labels = [a for a, _ in vals]
    heights = [v for _, v in vals]
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    ax.bar(labels, heights, color="tab:blue", alpha=0.8)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by arm")
    ax.set_ylim(0, max(heights) * 1.15 or 1.0)
    for i, h in enumerate(heights):
        ax.text(i, h, _fmt(h), ha="center", va="bottom", fontsize=8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)
