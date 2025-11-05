"""Python access to the SELPHI theory-of-mind research package."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

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


_base_dir = Path(__file__).resolve().parents[2] / "theory-of-mind" / "selphi"
_code_dir = _base_dir / "code"

_code_module = _load_package(__name__ + ".code", _code_dir, aliases=("selphi",))

if hasattr(_code_module, "__all__"):
    __all__ = list(_code_module.__all__)
    globals().update({name: getattr(_code_module, name) for name in _code_module.__all__})
else:
    exported = {name: value for name, value in vars(_code_module).items() if not name.startswith("_")}
    __all__ = list(exported)
    globals().update(exported)

if hasattr(_code_module, "__version__"):
    __version__ = _code_module.__version__
    if "__version__" not in __all__:
        __all__.append("__version__")

# Register key modules for both names
for module_name in ["benchmarks", "evals", "scenarios", "tasks"]:
    module_path = _code_dir / f"{module_name}.py"
    if module_path.exists():
        spec = spec_from_file_location(__name__ + f".{module_name}", module_path)
        if spec and spec.loader:
            module = module_from_spec(spec)
            sys.modules[__name__ + f".{module_name}"] = module
            sys.modules[f"selphi.{module_name}"] = module
            spec.loader.exec_module(module)
            globals()[module_name] = module
            if module_name not in __all__:
                __all__.append(module_name)
