"""Python access to the multi-agent communication research package."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

__all__ = []


def _load_package(name: str, directory: Path) -> ModuleType:
    """Load a package from an arbitrary directory and register it."""
    init_file = directory / "__init__.py"
    if not init_file.exists():
        raise ImportError(f"Cannot load package {name!r}; missing {init_file}.")

    spec = spec_from_file_location(name, init_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load package {name!r}; invalid spec for {init_file}.")

    module = module_from_spec(spec)
    # Ensure submodule imports (e.g., `.strategies`) resolve correctly
    spec.submodule_search_locations = [str(directory)]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The historical repository layout keeps the "real" multi-agent package in the
# `communication/multi-agent/multi_agent` directory (note the dash vs underscore).
# Earlier automation expected a `code/` directory but the repo never provided
# one, causing the import hook to fail and the entire subsystem to be
# unavailable.  The tests exercise the public `communication.multi_agent`
# namespace, so we resolve the actual on-disk location here instead of the
# non-existent legacy path.
_base_dir = Path(__file__).resolve().parent.parent / "multi-agent"
_code_dir = _base_dir / "multi_agent"

_code_module = _load_package(__name__ + ".code", _code_dir)

# Re-export symbols from the underlying code package for convenience
if hasattr(_code_module, "__all__"):
    __all__ = list(_code_module.__all__)
    globals().update({name: getattr(_code_module, name) for name in _code_module.__all__})
else:
    exported = {name: value for name, value in vars(_code_module).items() if not name.startswith("_")}
    __all__ = list(exported)
    globals().update(exported)

# Register important subpackages (crit, etc.) under the new namespace
for subpkg in ["crit"]:
    sub_dir = _code_dir / subpkg
    if sub_dir.exists():
        _load_package(__name__ + f".{subpkg}", sub_dir)
