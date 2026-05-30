#!/usr/bin/env python3
"""
Scaffold a new Hidden Layer research project with the lab conventions baked in.

Generates a project that is consistent, importable, offline-testable, and CI-wired from
the first commit — so new work starts at the fast path instead of re-deriving structure.

What it creates (for ``area`` / ``name``)::

    <area-dash>/<name-dash>/            # project files (dash-named)
      README.md  CLAUDE.md  hypotheses.md
      code/__init__.py  code/core.py    # importable code
      tests/conftest.py  tests/test_<name>.py   # offline smoke test (sim provider)
    <area_under>/<name_under>/__init__.py        # importable wrapper -> code/

It also (unless --no-makefile) adds the new test dir to OFFLINE_TESTS and the project to
LINT_PATHS in the root Makefile, so ``make ci`` runs it.

Usage::

    python scripts/new_project.py theory-of-mind working-memory
    python scripts/new_project.py communication swarm-signaling --question "Do ants..."

Naming: pass either dash or underscore form; both are derived. The area's underscore
package (e.g. theory_of_mind/) must be in setup.py's find_packages include list — for a
brand-new area the script warns you to add it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


def to_dash(s: str) -> str:
    return s.strip().replace("_", "-")


def to_under(s: str) -> str:
    return s.strip().replace("-", "_")


# --------------------------------------------------------------------- templates


def _readme(area_dash: str, name_dash: str, name_under: str, area_under: str, question: str) -> str:
    return f"""# {name_dash}

> {question}

Part of the **{area_dash}** research area. Importable as `{area_under}.{name_under}`.

## Quick start

```bash
# Offline smoke test (no API keys, no network)
python -m pytest {area_dash}/{name_dash}/tests -q
```

```python
from {area_under}.{name_under} import describe, echo_via_sim

print(describe())
print(echo_via_sim("hello"))   # runs through the harness 'sim' provider, fully offline
```

Swap in a real model by passing a provider to the harness (`provider="anthropic"`,
`"ollama"`, `"mlx"`, ...) — the same code path runs.

## Layout

```
{area_dash}/{name_dash}/
├── code/            # importable package (loaded as {area_under}.{name_under})
│   ├── core.py
│   └── __init__.py
├── tests/           # offline tests (sim provider); run in `make ci`
├── hypotheses.md    # hypothesis -> experiment -> result index
├── README.md
└── CLAUDE.md
```

See `CLAUDE.md` for the development guide.
"""


def _claude(area_dash: str, name_dash: str, name_under: str, area_under: str, question: str) -> str:
    return f"""# {name_dash} — Development Guide

## Project Overview

**Research Question**: {question}

**Uses**: `harness/` for the LLM provider abstraction (including the offline `sim`
provider), experiment tracking, evals, and the config-driven experiment runner.

## Architecture

```
code/
  core.py        # project logic (start here)
  __init__.py    # public API (re-exported as {area_under}.{name_under})
tests/           # offline, deterministic (sim provider)
```

## Conventions baked in

1. **Offline-first** — tests and demos run via `provider="sim"` (no API keys, no
   network), so they pass in CI. Drop in a real provider for real runs.
2. **Reproducible & logged** — use `harness.run_experiment` / `harness.get_tracker`
   so runs land in `experiments/` and can be tabulated by `harness.analysis`.
3. **Importable** — code lives in `code/` and is exposed as `{area_under}.{name_under}`
   via the wrapper, per the lab's dual-directory convention.

## Running an experiment

```python
from harness import ExperimentSpec, run_experiment, analysis

spec = ExperimentSpec.from_dict({{
    "name": "{name_under}_v1",
    "hypothesis": "...",
    "eval_type": "keyword",
    "dataset": {{"tasks": [{{"input": "...", "expected": "..."}}]}},
    "arms": [{{"name": "baseline", "provider": "sim", "params": {{"sim_response": "..."}}}}],
}})
report = run_experiment(spec)
print(analysis.to_markdown(report.to_rows()))
```

## Binding results into a paper

Add a generator to `papers/generate_results.py` that runs your experiment and emits
`papers/generated/{name_under}_*.tex` via `analysis.latex_macros` / `analysis.to_latex`,
then `\\input` those in `papers/{name_dash}.tex`. See `papers/README.md`.

## Testing

```bash
python -m pytest {area_dash}/{name_dash}/tests -q     # this project
make ci                                               # whole offline pipeline
```
"""


def _hypotheses(name_dash: str, question: str) -> str:
    return f"""# {name_dash} — Hypotheses

A living index linking each hypothesis to the experiment that tests it and the result.
Keep this current; it is the project's research audit trail.

**Driving question:** {question}

| ID | Hypothesis | Experiment (spec / run dir) | Result | Status |
|----|------------|-----------------------------|--------|--------|
| H1 | _state a falsifiable claim_ | _e.g. experiments/{name_dash}_v1.*_ | _ECE/acc/..._ | 🔬 open |

Status legend: 🔬 open · ✅ supported · ❌ refuted · ⏸ paused
"""


def _code_init(name_under: str, name_dash: str) -> str:
    return f'''"""
{name_dash} — research package.

Re-exports the public API from :mod:`core`. Imported as
``{name_under}`` (and ``{{area}}.{name_under}`` via the wrapper).
"""

from .core import describe, echo_via_sim

__version__ = "0.1.0"

__all__ = ["describe", "echo_via_sim"]
'''.replace("{area}", "{area_under}")


def _core(name_under: str, name_dash: str, area_dash: str) -> str:
    return f'''"""
Core logic for {name_dash}.

Starter module demonstrating the lab conventions: a pure, deterministic function and an
offline harness call via the ``sim`` provider. Replace these with your project's logic.
Keep heavy/optional imports (harness, numpy, ...) inside functions so the package imports
cheaply and offline.
"""

from __future__ import annotations

from typing import Dict

PROJECT = "{name_under}"
AREA = "{area_dash}"


def describe() -> Dict[str, str]:
    """Return basic project metadata (placeholder — replace with real logic)."""
    return {{"project": PROJECT, "area": AREA, "status": "scaffold"}}


def echo_via_sim(prompt: str) -> str:
    """
    Run a prompt through the harness ``sim`` provider (deterministic, offline).

    This is a template for how this project should call models: provider-agnostic, with
    ``sim`` as the offline/CI default. Swap ``provider`` for a real backend for real runs.
    """
    from harness import llm_call

    response = llm_call(prompt, provider="sim", sim_response=f"[{{PROJECT}}] {{prompt}}")
    return response.text
'''


def _conftest() -> str:
    return '''"""Make the repo root importable so this suite runs under any pytest invocation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
'''


def _test(name_under: str, area_under: str) -> str:
    return f'''"""Offline smoke tests for {name_under} (sim provider; no API keys, no network)."""

from {area_under}.{name_under} import describe, echo_via_sim


def test_describe():
    info = describe()
    assert info["project"] == "{name_under}"
    assert info["status"] == "scaffold"


def test_echo_via_sim_is_offline_and_deterministic():
    out = echo_via_sim("hello")
    assert "{name_under}" in out and "hello" in out
    # Deterministic: same input -> same output.
    assert echo_via_sim("hello") == out
'''


def _wrapper(area_dash: str, name_dash: str) -> str:
    return f'''"""Python access to the {name_dash} research package.

Loads the implementation from the dash-named project directory
(``{area_dash}/{name_dash}/code``) so it is importable via this underscore package,
mirroring the lab's dual-directory convention.
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

__all__: list = []


def _load_package(name: str, directory: Path, aliases: tuple = ()) -> ModuleType:
    init_file = directory / "__init__.py"
    if not init_file.exists():
        raise ImportError(f"Cannot load package {{name!r}}; missing {{init_file}}.")
    spec = spec_from_file_location(name, init_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load package {{name!r}}; invalid spec for {{init_file}}.")
    module = module_from_spec(spec)
    spec.submodule_search_locations = [str(directory)]
    sys.modules[name] = module
    for alias in aliases:
        sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_code_dir = Path(__file__).resolve().parents[2] / "{area_dash}" / "{name_dash}" / "code"
_code_module = _load_package(__name__ + ".code", _code_dir)

if hasattr(_code_module, "__all__"):
    __all__ = list(_code_module.__all__)
    globals().update({{name: getattr(_code_module, name) for name in _code_module.__all__}})
else:
    exported = {{name: value for name, value in vars(_code_module).items() if not name.startswith("_")}}
    __all__ = list(exported)
    globals().update(exported)
'''


# ------------------------------------------------------------- makefile wiring


def update_make_var(text: str, var: str, value: str) -> str:
    """Append ``value`` to a ``VAR := a b c`` line if not already present. Pure function."""
    pattern = re.compile(rf"^({re.escape(var)}\s*:=\s*)(.*)$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return text  # var not found; caller may warn
    existing = m.group(2).split()
    if value in existing:
        return text
    return pattern.sub(lambda mm: mm.group(1) + (mm.group(2) + " " + value).strip(), text, count=1)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def create_project(
    area: str,
    name: str,
    root: Path = REPO_ROOT,
    question: str = "TODO: state the research question.",
    update_makefile: bool = True,
) -> List[Path]:
    """Scaffold a project. Returns the list of files written."""
    area_dash, area_under = to_dash(area), to_under(area)
    name_dash, name_under = to_dash(name), to_under(name)

    proj = root / area_dash / name_dash
    if proj.exists():
        raise SystemExit(f"Refusing to overwrite existing project: {proj}")

    written: List[Path] = []

    def emit(rel: Path, content: str) -> None:
        _write(rel, content)
        written.append(rel)

    # Project files (dash dir)
    emit(proj / "README.md", _readme(area_dash, name_dash, name_under, area_under, question))
    emit(proj / "CLAUDE.md", _claude(area_dash, name_dash, name_under, area_under, question))
    emit(proj / "hypotheses.md", _hypotheses(name_dash, question))
    emit(proj / "code" / "__init__.py", _code_init(name_under, name_dash).replace("{area_under}", area_under))
    emit(proj / "code" / "core.py", _core(name_under, name_dash, area_dash))
    emit(proj / "tests" / "conftest.py", _conftest())
    emit(proj / "tests" / f"test_{name_under}.py", _test(name_under, area_under))

    # Importable wrapper (underscore package). Ensure the area package exists.
    area_pkg_dir = root / area_under
    area_init = area_pkg_dir / "__init__.py"
    if not area_init.exists():
        emit(area_init, f'"""{area_dash} research area packages."""\n')
    emit(area_pkg_dir / name_under / "__init__.py", _wrapper(area_dash, name_dash))

    # Wire into the offline CI scopes.
    if update_makefile:
        mk = root / "Makefile"
        if mk.exists():
            text = mk.read_text()
            text = update_make_var(text, "OFFLINE_TESTS", f"{area_dash}/{name_dash}/tests")
            text = update_make_var(text, "LINT_PATHS", f"{area_dash}/{name_dash}")
            mk.write_text(text)

    return written


def _maybe_warn_setup(area_under: str) -> None:
    setup = REPO_ROOT / "setup.py"
    if setup.exists() and f'"{area_under}"' not in setup.read_text():
        print(
            f"\n⚠️  New area '{area_under}' is not in setup.py find_packages include list.\n"
            f'   Add "{area_under}" and "{area_under}.*" there so it installs with `pip install -e .`.'
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a new Hidden Layer project.")
    parser.add_argument("area", help="Research area, e.g. theory-of-mind, communication, alignment")
    parser.add_argument("name", help="Project name, e.g. working-memory")
    parser.add_argument("--question", default="TODO: state the research question.", help="Driving research question")
    parser.add_argument("--no-makefile", action="store_true", help="Do not edit the root Makefile")
    args = parser.parse_args(argv)

    written = create_project(args.area, args.name, question=args.question, update_makefile=not args.no_makefile)
    print(f"Scaffolded {to_dash(args.area)}/{to_dash(args.name)} ({len(written)} files):")
    for p in written:
        print(f"  {p.relative_to(REPO_ROOT)}")
    _maybe_warn_setup(to_under(args.area))
    print("\nNext steps:")
    print(f"  python -m pytest {to_dash(args.area)}/{to_dash(args.name)}/tests -q")
    print("  make ci")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
