from __future__ import annotations

import importlib
import json
from typing import Any

get_root_columns = importlib.import_module(
    "src.pipeline.Progressive-Pipeline-Curation.feature-Based_step1"
).get_root_columns


def _roots_for_column(col: str) -> list[str]:
    return list(dict.fromkeys(get_root_columns(col)))


def _step_key(step: dict[str, Any]) -> tuple[str, str]:
    return (
        step.get("transformation", ""),
        json.dumps(step.get("params") or {}, sort_keys=True),
    )
