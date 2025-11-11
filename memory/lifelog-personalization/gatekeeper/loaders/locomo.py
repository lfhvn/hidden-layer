"""LoCoMo dataset loader.

The LoCoMo benchmark is released as JSON structures containing multi-session
conversations grounded in lifelog event graphs. The loader below normalises the
sessions into a turn-level dataframe while preserving metadata required by the
scorers (timestamps, event ids, preference state, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DatasetLoader, SplitSpec, ensure_columns


@dataclass
class LoCoMoTurn:
    """Representation of a single conversational turn."""

    conversation_id: str
    turn_index: int
    speaker: str
    text: str
    timestamp: Optional[str]
    events: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    reference: Optional[Dict[str, Any]]

    def to_row(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "utterance": self.text,
            "timestamp": self.timestamp,
            "events": self.events,
            "preferences": self.preferences,
            "reference": self.reference,
        }


class LoCoMoDatasetLoader(DatasetLoader):
    """Loader for the LoCoMo benchmark and related long-context datasets."""

    def __init__(self, dataset_name: str, cache_dir: Path, data_root: Path) -> None:
        super().__init__(dataset_name, cache_dir, data_root)

    def _split_path(self, split: SplitSpec) -> Optional[Path]:
        base = self.raw_root()
        candidates = [
            base / f"{split.name}.jsonl",
            base / f"{split.name}.json",
            base / f"{split.name}.ndjson",
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
                f"LoCoMo split '{split.name}' not found. Expected one of: JSONL/JSON files under {self.raw_root()}"
            )

        records = self._read_json_records(path)
        rows: List[Dict[str, Any]] = []
        for record in records:
            conv_id = str(record.get("conversation_id") or record.get("id") or record.get("session_id"))
            preference_state = record.get("preferences", {})
            reference = record.get("reference") or record.get("targets")
            turns = record.get("turns") or record.get("dialogue") or []
            for turn_idx, turn in enumerate(turns):
                locomo_turn = LoCoMoTurn(
                    conversation_id=conv_id,
                    turn_index=turn_idx,
                    speaker=str(turn.get("speaker", "unknown")),
                    text=str(turn.get("text") or turn.get("utterance") or ""),
                    timestamp=turn.get("timestamp") or record.get("timestamp"),
                    events=turn.get("events") or record.get("events") or [],
                    preferences=turn.get("preferences") or preference_state,
                    reference=self._turn_reference(turn, reference),
                )
                rows.append(locomo_turn.to_row())

        df = pd.DataFrame(rows)
        required = ["conversation_id", "turn_index", "utterance"]
        return ensure_columns(df, required)

    def _read_json_records(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix in {".json", ""}:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data.get("data") or data.get("sessions") or [data]
            return data

        records: List[Dict[str, Any]] = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _turn_reference(self, turn: Dict[str, Any], conversation_reference: Any) -> Optional[Dict[str, Any]]:
        if isinstance(turn.get("reference"), dict):
            return turn["reference"]
        if isinstance(conversation_reference, dict):
            return conversation_reference
        return None
