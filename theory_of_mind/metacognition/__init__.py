"""Python access to the metacognition research package.

Loads the implementation that lives in the dash-named project directory
(``theory-of-mind/metacognition/code``) so it is importable as
``theory_of_mind.metacognition`` (and as ``metacognition``), mirroring the
dual-directory convention used elsewhere in the lab.
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

__all__ = []


def _load_package(name: str, directory: Path, aliases: tuple[str, ...] = ()) -> ModuleType:
    init_file = directory / "__init__.py"
    if not init_file.exists():
        raise ImportError(f"Cannot load package {name!r}; missing {init_file}.")

    spec = spec_from_file_location(name, init_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load package {name!r}; invalid spec for {init_file}.")

    module = module_from_spec(spec)
    spec.submodule_search_locations = [str(directory)]
    sys.modules[name] = module
    for alias in aliases:
        sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_base_dir = Path(__file__).resolve().parents[2] / "theory-of-mind" / "metacognition"
_code_dir = _base_dir / "code"

_code_module = _load_package(__name__ + ".code", _code_dir, aliases=("metacognition",))

if hasattr(_code_module, "__all__"):
    __all__ = list(_code_module.__all__)
    globals().update({name: getattr(_code_module, name) for name in _code_module.__all__})
else:
    exported = {name: value for name, value in vars(_code_module).items() if not name.startswith("_")}
    __all__ = list(exported)
    globals().update(exported)
