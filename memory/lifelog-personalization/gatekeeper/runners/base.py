"""Common runner utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from ..datasets.registry import get_dataset_loader
from ..loaders.base import SplitSpec


@dataclass
class LoadedDataset:
    name: str
    split: str
    dataframe: pd.DataFrame


@dataclass
class EvaluationResult:
    name: str
    metrics: Dict[str, float]


def load_config(path: Path) -> Dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def load_configured_datasets(config: Dict) -> List[LoadedDataset]:
    datasets_cfg = config.get("datasets", [])
    loaded: List[LoadedDataset] = []
    for dataset_cfg in datasets_cfg:
        name = dataset_cfg["name"]
        splits = dataset_cfg.get("splits", ["test"])
        loader = get_dataset_loader(name)
        for split in splits:
            df = loader.load(SplitSpec(split))
            loaded.append(LoadedDataset(name=name, split=split, dataframe=df))
    return loaded
