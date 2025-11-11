"""Generic fall-back dataset loader.

This loader handles JSON/JSONL/CSV files by exposing them as dataframes without
any additional normalisation. It is useful for auxiliary TTL/TTT corpora used to
measure domain-shift adaptation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DatasetLoader, SplitSpec


class GenericDatasetLoader(DatasetLoader):
    """Fallback loader for structured tabular/text corpora."""

    def __init__(self, dataset_name: str, cache_dir: Path, data_root: Path) -> None:
        super().__init__(dataset_name, cache_dir, data_root)

    def _split_path(self, split: SplitSpec) -> Optional[Path]:
        base = self.raw_root()
        candidates = [
            base / f"{split.name}.jsonl",
            base / f"{split.name}.json",
            base / f"{split.name}.csv",
            base / f"{split.name}.tsv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_to_dataframe(self, split: SplitSpec) -> pd.DataFrame:
        path = split.path or self._split_path(split)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Generic dataset '{self.dataset_name}' split '{split.name}' not found under {self.raw_root()}"
            )

        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            return pd.DataFrame(data)

        records: List[Dict[str, Any]] = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return pd.DataFrame(records)
