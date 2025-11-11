"""LaMP / PrefEval / PersonaMem dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .base import DatasetLoader, SplitSpec, ensure_columns


class LaMPDatasetLoader(DatasetLoader):
    """Loader that handles preference-centric benchmarks with shared schema."""

    def __init__(self, dataset_name: str, cache_dir: Path, data_root: Path) -> None:
        super().__init__(dataset_name, cache_dir, data_root)

    def _split_path(self, split: SplitSpec) -> Optional[Path]:
        base = self.raw_root()
        candidates = [
            base / f"{split.name}.jsonl",
            base / f"{split.name}.json",
            base / f"{split.name}.parquet",
            base / split.name / "data.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_to_dataframe(self, split: SplitSpec) -> pd.DataFrame:
        path = split.path or self._split_path(split)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Dataset '{self.dataset_name}' split '{split.name}' not found under {self.raw_root()}"
            )

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            records = self._read_records(path)
            df = pd.DataFrame(records)

        # Normalise columns.
        if "input" in df.columns and "query" not in df.columns:
            df = df.rename(columns={"input": "query"})
        if "output" in df.columns and "answer" not in df.columns:
            df = df.rename(columns={"output": "answer"})
        if "preference" in df.columns and "preferences" not in df.columns:
            df = df.rename(columns={"preference": "preferences"})

        required = ["query"]
        df = ensure_columns(df, required)
        df.setdefault("preferences", [{}] * len(df))
        df.setdefault("answer", [None] * len(df))
        df.setdefault("session_id", [None] * len(df))
        return df

    def _read_records(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data.get("data") or data.get("examples") or [data]
            return data

        records: List[Dict[str, Any]] = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
