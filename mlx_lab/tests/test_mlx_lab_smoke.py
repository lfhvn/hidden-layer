"""Offline smoke tests for the MLX Lab CLI utilities (no model load, no network)."""

import mlx_lab


def test_public_classes_exist():
    for cls in ("ModelManager", "ConfigManager", "PerformanceBenchmark", "ConceptBrowser"):
        assert hasattr(mlx_lab, cls)


def test_config_manager_reads_current_config_offline():
    cm = mlx_lab.ConfigManager()
    cfg = cm.get_current_config()
    assert isinstance(cfg, dict)
