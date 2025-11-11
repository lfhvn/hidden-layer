"""Python access to the lifelog personalization gatekeeper package."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

__all__ = ["gatekeeper"]


def _load_package(name: str, directory: Path) -> ModuleType:
    """Load a package from an arbitrary directory and register it."""
    init_file = directory / "__init__.py"
    if not init_file.exists():
        raise ImportError(f"Cannot load package {name!r}; missing {init_file}.")

    spec = spec_from_file_location(name, init_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load package {name!r}; invalid spec for {init_file}.")

    module = module_from_spec(spec)
    spec.submodule_search_locations = [str(directory)]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base_dir = Path(__file__).resolve().parent / "lifelog-personalization"
_gatekeeper_dir = _base_dir / "gatekeeper"

_gatekeeper_module = _load_package(__name__ + ".gatekeeper", _gatekeeper_dir)

# Re-export contents for convenience
if hasattr(_gatekeeper_module, "__all__"):
    __all__ = list(_gatekeeper_module.__all__)
    globals().update({name: getattr(_gatekeeper_module, name) for name in _gatekeeper_module.__all__})
else:
    exported = {name: value for name, value in vars(_gatekeeper_module).items() if not name.startswith("_")}
    __all__ = list(exported)
    globals().update(exported)


gatekeeper = _gatekeeper_module
__all__.append("gatekeeper")
