from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

OUTPUTS_FILE = "outputs.jsonl"
PIPELINES_FILE = "pipelines.jsonl"
OUTPUT_TAGS_FILE = "output_tags.jsonl"


def _pipeline_record(state) -> dict[str, Any]:
    phase1 = []
    phase2 = []
    phase3 = []

    for step in state.steps:
        phase = step.get("phase")
        if phase == "phase1_structural":
            phase1.append(step)
        elif phase == "phase2_cell_level":
            phase2.append(step)
        elif phase == "phase3_structural":
            phase3.append(step)

    return {
        "phase1_structural": phase1,
        "phase2_cell_level": phase2,
        "phase3_structural": phase3,
        "phase2_id": getattr(state, "phase2_id", None),
    }


def _table_record(df: pd.DataFrame) -> dict[str, Any]:
    return {"csv": df.to_csv(index=False)}


def _append_jsonl(path: str, obj: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


class OutputStore:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.outputs_path = os.path.join(folder_path, OUTPUTS_FILE)
        self.pipelines_path = os.path.join(folder_path, PIPELINES_FILE)
        self.tags_path = os.path.join(folder_path, OUTPUT_TAGS_FILE)
        self._count = 0

    def begin(self) -> None:
        for path in (
            self.outputs_path,
            self.pipelines_path,
            self.tags_path,
        ):
            if os.path.exists(path):
                os.remove(path)

    def append(self, idx: int, state) -> None:
        _append_jsonl(self.outputs_path, {"idx": idx, "table": _table_record(state.df)})
        _append_jsonl(self.pipelines_path, {"idx": idx, "pipeline": _pipeline_record(state)})
        _append_jsonl(self.tags_path, {"idx": idx, "tags": state.tags})
        self._count += 1

    @property
    def count(self) -> int:
        return self._count


