"""Dataset registry for evaluation harness."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Type

from ..loaders.base import DatasetLoader
from ..loaders.generic import GenericDatasetLoader
from ..loaders.lamp import LaMPDatasetLoader
from ..loaders.locomo import LoCoMoDatasetLoader
from ..loaders.ntcir18 import NTCIR18DatasetLoader

DATASET_LOADERS: Dict[str, Type[DatasetLoader]] = {
    "lamp": LaMPDatasetLoader,
    "prefeval": LaMPDatasetLoader,  # re-uses same format with preference metadata
    "personamem": LaMPDatasetLoader,
    "locomo": LoCoMoDatasetLoader,
    "lsc24": NTCIR18DatasetLoader,
    "ntcir18_lsat": NTCIR18DatasetLoader,
    "ntcir18_lqat": NTCIR18DatasetLoader,
    "imageclef20": NTCIR18DatasetLoader,
    "longbench": GenericDatasetLoader,
    "infinibench": GenericDatasetLoader,
    "domain_shift_news_2025q3": GenericDatasetLoader,
    "legal_contracts_holdout": GenericDatasetLoader,
    "coding_stackexchange_recent": GenericDatasetLoader,
    "counterfact": GenericDatasetLoader,
    "zsre": GenericDatasetLoader,
}


def get_dataset_loader(dataset: str, cache_base: Optional[Path] = None, data_root: Optional[Path] = None) -> DatasetLoader:
    """Instantiate the loader for ``dataset``.

    Args:
        dataset: Dataset identifier from configuration files.
        cache_base: Optional override for the cache directory.
        data_root: Optional override for the raw dataset root.

    Returns:
        Instantiated loader.
    """

    if dataset not in DATASET_LOADERS:
        raise KeyError(f"Unknown dataset '{dataset}'. Known datasets: {sorted(DATASET_LOADERS.keys())}")

    loader_cls = DATASET_LOADERS[dataset]

    base_cache = cache_base or Path(os.environ.get("EVAL_CACHE_DIR", "eval/.cache"))
    cache_dir = base_cache / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_data_root = data_root or Path(os.environ.get("EVAL_DATA_ROOT", "eval/datasets"))
    return loader_cls(dataset_name=dataset, cache_dir=cache_dir, data_root=base_data_root)
