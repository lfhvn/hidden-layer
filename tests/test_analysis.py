"""Tests for the analysis / reporting layer (harness.analysis)."""

from pathlib import Path

import pytest

from harness import ExperimentSpec, analysis, run_experiment

ROWS = [
    {"arm": "oracle", "provider": "sim", "model": "m", "strategy": "single", "n": 3, "accuracy": 1.0},
    {"arm": "gpt-4o", "provider": "openai", "model": "gpt-4o", "strategy": "single", "n": 3, "accuracy": 0.5},
]


def test_to_markdown_has_header_and_rows():
    md = analysis.to_markdown(ROWS)
    assert md.startswith("| arm |")
    assert "oracle" in md and "gpt-4o" in md
    # metric column appears after meta columns
    assert "accuracy" in md.splitlines()[0]


def test_to_csv_roundtrips_columns():
    csv_text = analysis.to_csv(ROWS, columns=["arm", "accuracy"])
    lines = csv_text.splitlines()
    assert lines[0] == "arm,accuracy"
    assert "oracle,1" in csv_text


def test_to_latex_is_tabular_and_escapes():
    tex = analysis.to_latex(ROWS, caption="Acc", label="tab:acc")
    assert r"\begin{tabular}" in tex and r"\toprule" in tex
    assert r"gpt-4o".replace("_", r"\_") in tex or "gpt-4o" in tex
    assert r"\caption{Acc}" in tex and r"\label{tab:acc}" in tex


def test_latex_macros_naming():
    macros = analysis.latex_macros(ROWS, metrics=["accuracy"])
    # 'gpt-4o' -> GptFouro (digits become words, '-' stripped, letters only).
    assert r"\newcommand{\resultOracleAccuracy}{1}" in macros
    assert r"\resultGptFouroAccuracy" in macros


def test_format_rows_dispatch_and_unknown():
    assert analysis.format_rows(ROWS, "markdown").startswith("|")
    assert "tabular" in analysis.format_rows(ROWS, "latex")
    assert analysis.format_rows(ROWS, "csv").startswith("arm")
    with pytest.raises(ValueError):
        analysis.format_rows(ROWS, "pdf")


def test_empty_rows_are_safe():
    assert analysis.to_markdown([]) == "_(no runs)_"
    assert analysis.to_csv([]) == ""
    assert "no runs" in analysis.to_latex([])


def test_ascii_bar_renders():
    out = analysis._ascii_bar(ROWS, "accuracy")
    assert "oracle" in out and "#" in out


def test_load_runs_roundtrip(tmp_path):
    spec = ExperimentSpec.from_dict(
        {
            "name": "roundtrip",
            "eval_type": "keyword",
            "dataset": {"tasks": [{"input": "Capital of France?", "expected": "paris"}]},
            "arms": [
                {"name": "oracle", "provider": "sim", "params": {"sim_response": "paris"}},
                {"name": "wrong", "provider": "sim", "params": {"sim_response": "nope"}},
            ],
        }
    )
    run_experiment(spec, base_dir=str(tmp_path), track=True)

    runs = analysis.load_runs(str(tmp_path))
    assert len(runs) == 2
    rows = analysis.runs_to_rows(runs)
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["oracle"]["accuracy"] == pytest.approx(1.0)
    assert by_arm["wrong"]["accuracy"] == pytest.approx(0.0)
    # arm label is derived from experiment_name 'roundtrip.<arm>'
    assert set(by_arm) == {"oracle", "wrong"}


def test_load_runs_glob_pattern(tmp_path):
    spec = ExperimentSpec.from_dict(
        {
            "name": "globme",
            "eval_type": "keyword",
            "dataset": {"tasks": [{"input": "Capital of France?", "expected": "paris"}]},
            "arms": [{"name": "a", "provider": "sim", "params": {"sim_response": "paris"}}],
        }
    )
    run_experiment(spec, base_dir=str(tmp_path), track=True)
    runs = analysis.load_runs(str(Path(tmp_path) / "globme.*"))
    assert len(runs) == 1
