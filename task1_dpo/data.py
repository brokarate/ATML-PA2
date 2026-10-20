from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


PROMPT_KEYS = ("prompt", "instruction", "query", "input")
CHOSEN_KEYS = ("chosen", "preferred", "response_chosen", "y_plus", "winner")
REJECTED_KEYS = ("rejected", "response_rejected", "y_minus", "loser")


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, list):
                return "\n".join(str(item) for item in value)
            return str(value)
    raise KeyError(f"None of these keys were found: {keys}")


def normalize_preference_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "prompt": _first_present(row, PROMPT_KEYS),
        "chosen": _first_present(row, CHOSEN_KEYS),
        "rejected": _first_present(row, REJECTED_KEYS),
    }
    for key in ("id", "prompt_id", "stratum", "category"):
        if key in row:
            normalized[key] = row[key]
    return normalized


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


class PreferenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = [normalize_preference_row(row) for row in load_jsonl(self.path)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def discover_preference_file(config_path: str | Path, fallback_patterns: list[str]) -> Path:
    configured = Path(config_path)
    if configured.exists():
        return configured

    for pattern in fallback_patterns:
        matches = sorted(Path(".").glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find preference file '{configured}'. Download course assets or update the config path."
    )

