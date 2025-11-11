"""Dataset loader abstractions for evaluation harness."""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


@dataclass
class SplitSpec:
    """Specification for a dataset split.

    Attributes:
        name: Name of the split (e.g., "train", "dev", "test").
        path: Optional explicit path override. When omitted the loader determines the
            path from the dataset root.
    """

    name: str
    path: Optional[Path] = None


class DatasetLoader(abc.ABC):
    """Base class for dataset adapters.

    Loaders are responsible for reading benchmark artifacts, normalising them into a
    columnar representation, and persisting cached parquet files for faster reuse.
    """

    dataset_name: str
    cache_dir: Path
    data_root: Path

    def __init__(self, dataset_name: str, cache_dir: Path, data_root: Path) -> None:
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.data_root = data_root
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, split: SplitSpec) -> pd.DataFrame:
        """Return the normalised dataframe for ``split``.

        The loader writes a cached parquet file alongside the raw data. Subsequent
        calls reuse the cached version unless the raw data is newer or the cache is
        missing.
        """

        cache_path = self.cache_dir / f"{split.name}.parquet"
        raw_timestamp = self._raw_timestamp(split)
        if cache_path.exists() and raw_timestamp is not None and cache_path.stat().st_mtime >= raw_timestamp:
            return pd.read_parquet(cache_path)

        df = self._load_to_dataframe(split)
        df.to_parquet(cache_path, index=False)
        metadata = {"dataset": self.dataset_name, "split": split.name, "rows": len(df)}
        (self.cache_dir / f"{split.name}.meta.json").write_text(json.dumps(metadata, indent=2))
        return df

    @abc.abstractmethod
    def _load_to_dataframe(self, split: SplitSpec) -> pd.DataFrame:
        """Load ``split`` and return a normalised dataframe.

        Implementations should produce the following canonical columns when
        applicable:
            * ``query_id`` – unique identifier for the evaluation topic/query.
            * ``query`` – textual description of the information need.
            * ``context`` – supporting context (captions, metadata, etc.).
            * ``answer`` – reference answers for QA-style tasks.
            * ``timestamp`` – integer or ISO-8601 timestamp for temporal metrics.
            * ``entities`` – structured entity annotations (dict or list).
        """

    def raw_root(self) -> Path:
        """Return the directory containing the raw dataset files.

        Subclasses may override this when they require a different directory
        structure, but by default the dataset root is ``self.data_root /
        self.dataset_name``.
        """

        return self.data_root / self.dataset_name

    def _raw_timestamp(self, split: SplitSpec) -> Optional[float]:
        path = split.path or self._split_path(split)
        if path is None or not path.exists():
            return None
        return path.stat().st_mtime

    def _split_path(self, split: SplitSpec) -> Optional[Path]:
        """Return the raw path for the split, if well-known."""
        return None


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required columns {missing}. Present columns: {sorted(df.columns)}"
        )
    return df


def concatenate_loaders(loaders: Mapping[str, DatasetLoader], splits: Mapping[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """Load a collection of datasets and splits.

    Args:
        loaders: Mapping from dataset name to loader instance.
        splits: Mapping from dataset name to the splits that should be loaded.

    Returns:
        Dictionary keyed by ``"{dataset}:{split}"`` containing dataframes.
    """

    output: Dict[str, pd.DataFrame] = {}
    for dataset_name, split_names in splits.items():
        loader = loaders[dataset_name]
        for split_name in split_names:
            df = loader.load(SplitSpec(split_name))
            output[f"{dataset_name}:{split_name}"] = df
    return output
