"""Tests for the project scaffold (scripts/new_project.py)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import new_project as np  # noqa: E402  (import follows sys.path setup)


def test_update_make_var_appends_and_is_idempotent():
    text = "OFFLINE_TESTS := tests foo/bar\nLINT_PATHS := harness\n"
    out = np.update_make_var(text, "OFFLINE_TESTS", "baz/tests")
    assert "OFFLINE_TESTS := tests foo/bar baz/tests" in out
    # Re-applying changes nothing.
    assert np.update_make_var(out, "OFFLINE_TESTS", "baz/tests") == out
    # A var that isn't present leaves the text untouched.
    assert np.update_make_var(text, "DOES_NOT_EXIST", "x") == text


def test_create_project_structure_and_valid_python(tmp_path):
    written = np.create_project("demo-area", "demo-project", root=tmp_path, update_makefile=False)

    proj = tmp_path / "demo-area" / "demo-project"
    for rel in [
        "README.md",
        "CLAUDE.md",
        "hypotheses.md",
        "code/__init__.py",
        "code/core.py",
        "tests/conftest.py",
        "tests/test_demo_project.py",
    ]:
        assert (proj / rel).exists(), f"missing {rel}"

    # Importable wrapper + area package were created.
    assert (tmp_path / "demo_area" / "__init__.py").exists()
    assert (tmp_path / "demo_area" / "demo_project" / "__init__.py").exists()

    # Every generated Python file is syntactically valid.
    py_files = [p for p in written if p.suffix == ".py"]
    assert py_files
    for p in py_files:
        compile(p.read_text(), str(p), "exec")


def test_create_project_refuses_overwrite(tmp_path):
    np.create_project("area", "proj", root=tmp_path, update_makefile=False)
    with pytest.raises(SystemExit):
        np.create_project("area", "proj", root=tmp_path, update_makefile=False)


def test_create_project_wires_makefile(tmp_path):
    mk = tmp_path / "Makefile"
    mk.write_text("OFFLINE_TESTS := tests\nLINT_PATHS := harness\n")
    np.create_project("demo-area", "demo-project", root=tmp_path, update_makefile=True)
    text = mk.read_text()
    assert "demo-area/demo-project/tests" in text  # added to OFFLINE_TESTS
    assert "LINT_PATHS := harness demo-area/demo-project" in text


def test_naming_normalization(tmp_path):
    # Underscore input is normalized to the dash project dir + underscore package.
    np.create_project("demo_area", "demo_project", root=tmp_path, update_makefile=False)
    assert (tmp_path / "demo-area" / "demo-project" / "code" / "core.py").exists()
    assert (tmp_path / "demo_area" / "demo_project" / "__init__.py").exists()
