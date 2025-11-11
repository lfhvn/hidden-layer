"""Scorecard rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..runners.base import EvaluationResult


@dataclass
class Scorecard:
    title: str
    results: List[EvaluationResult]

    def to_markdown(self) -> str:
        lines = [f"# {self.title}"]
        for result in self.results:
            lines.append(f"\n## {result.name}")
            for metric, value in sorted(result.metrics.items()):
                lines.append(f"- **{metric}**: {value:.4f}")
        return "\n".join(lines)

    def to_html(self) -> str:
        sections = []
        for result in self.results:
            rows = "".join(
                f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>" for metric, value in sorted(result.metrics.items())
            )
            sections.append(f"<section><h2>{result.name}</h2><table>{rows}</table></section>")
        return f"<article><h1>{self.title}</h1>{''.join(sections)}</article>"

    def write(self, output: Path, fmt: str = "markdown") -> None:
        if fmt == "markdown":
            output.write_text(self.to_markdown())
        elif fmt == "html":
            output.write_text(self.to_html())
        else:
            raise ValueError(f"Unsupported format '{fmt}'")


def render_scorecard(results: Iterable[EvaluationResult], title: str, output: Path, fmt: str = "markdown") -> None:
    card = Scorecard(title=title, results=list(results))
    card.write(output, fmt=fmt)
