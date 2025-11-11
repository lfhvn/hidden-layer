"""NTCIR-18 / LSC style lifelog retrieval loader."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DatasetLoader, SplitSpec, ensure_columns


class NTCIR18DatasetLoader(DatasetLoader):
    """Loader for lifelog retrieval datasets with topic definitions."""

    TOPIC_COLUMNS = ["topic_id", "query", "time_range", "entities"]

    def __init__(self, dataset_name: str, cache_dir: Path, data_root: Path) -> None:
        super().__init__(dataset_name, cache_dir, data_root)

    def _split_path(self, split: SplitSpec) -> Optional[Path]:
        base = self.raw_root()
        candidates = [
            base / f"{split.name}.json",
            base / f"{split.name}.jsonl",
            base / f"{split.name}.tsv",
            base / "topics" / f"{split.name}.tsv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_to_dataframe(self, split: SplitSpec) -> pd.DataFrame:
        path = split.path or self._split_path(split)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Lifelog dataset '{self.dataset_name}' split '{split.name}' missing under {self.raw_root()}"
            )

        if path.suffix == ".tsv":
            df = self._read_topics_tsv(path)
        else:
            df = self._read_topics_json(path)

        df = ensure_columns(df, ["topic_id", "query"])
        df.setdefault("start_time", [None] * len(df))
        df.setdefault("end_time", [None] * len(df))
        df.setdefault("entities", [[]] * len(df))
        return df

    def _read_topics_tsv(self, path: Path) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows.append(
                    {
                        "topic_id": row.get("topic_id") or row.get("topic"),
                        "query": row.get("query") or row.get("description"),
                        "start_time": row.get("start_time") or row.get("start"),
                        "end_time": row.get("end_time") or row.get("end"),
                        "entities": self._parse_entities(row.get("entities")),
                    }
                )
        return pd.DataFrame(rows)

    def _read_topics_json(self, path: Path) -> pd.DataFrame:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            records = data.get("topics") or data.get("queries") or data.get("data") or []
        else:
            records = data

        rows: List[Dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "topic_id": record.get("topic_id") or record.get("topic") or record.get("id"),
                    "query": record.get("query") or record.get("description"),
                    "start_time": record.get("start_time") or record.get("start"),
                    "end_time": record.get("end_time") or record.get("end"),
                    "entities": record.get("entities") or [],
                }
            )
        return pd.DataFrame(rows)

    def _parse_entities(self, raw: Optional[str]) -> List[str]:
        if raw is None:
            return []
        raw = raw.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in raw.split(",") if part.strip()]
