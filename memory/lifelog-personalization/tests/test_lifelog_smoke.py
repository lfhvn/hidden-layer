"""Offline smoke tests for the lifelog-personalization gatekeeper.

Guards the importable wrapper (which was silently mis-pathed) and the numpy-free
submodules. numpy-dependent scorers are exercised only when numpy is available.
"""

import importlib

import pytest

import memory.lifelog_personalization as lifelog


def test_package_imports_via_wrapper():
    # Regression guard: the wrapper used to point at a non-existent nested dir.
    assert hasattr(lifelog, "gatekeeper")


@pytest.mark.parametrize("submodule", ["policies", "loaders", "runners"])
def test_numpy_free_submodules_import(submodule):
    mod = importlib.import_module(f"memory.lifelog_personalization.gatekeeper.{submodule}")
    assert mod is not None


def test_scorers_import_when_numpy_available():
    pytest.importorskip("numpy")
    temporal = importlib.import_module("memory.lifelog_personalization.gatekeeper.scorers.temporal")
    # Module exposes at least one public callable (a scorer).
    assert any(callable(getattr(temporal, n)) for n in dir(temporal) if not n.startswith("_"))
