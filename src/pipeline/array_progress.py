from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .array_config import get_array_id, get_num_arrays, project_root


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def run_progress_path() -> Path:
    return project_root() / "cache" / "run_progress.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


BUNDLE_FILES = ("outputs.jsonl", "pipelines.jsonl", "output_tags.jsonl")


def folder_has_jsonl(wiki_root: str, folder_name: str) -> bool:
    folder = Path(wiki_root) / folder_name
    return any((folder / name).is_file() for name in BUNDLE_FILES)


def scan_folders_with_jsonl(
    wiki_root: str, assigned_folders: list[str]
) -> list[str]:
    return [
        name
        for name in assigned_folders
        if folder_has_jsonl(wiki_root, name)
    ]


def write_array_progress(
    *,
    wiki_root: str,
    assigned_folders: list[str],
    completed_folders: list[str],
    failed_folders: list[str],
    skipped_folders: list[str],
    array_id: int | None = None,
    num_arrays: int | None = None,
) -> Path:
    array_id = get_array_id() if array_id is None else array_id
    num_arrays = get_num_arrays() if num_arrays is None else num_arrays
    from .array_config import manifest_folders_for_array

    manifest_assigned = manifest_folders_for_array(array_id, num_arrays)
    if manifest_assigned is not None:
        assigned_folders = list(manifest_assigned)
    assigned = len(assigned_folders)
    completed_set = set(completed_folders)
    failed_set = set(failed_folders)
    skipped_set = set(skipped_folders)
    completed_count = len(completed_folders)
    failed_count = len(failed_folders)
    skipped_count = len(skipped_folders)
    remaining_count = sum(
        1
        for name in assigned_folders
        if name not in completed_set
        and name not in failed_set
        and name not in skipped_set
    )

    payload: dict[str, Any] = {
        "array_id": array_id,
        "num_arrays": num_arrays,
        "wiki_root": str(Path(wiki_root).resolve()),
        "assigned": assigned,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "remaining_count": remaining_count,
        "completed_folders": sorted(completed_folders),
        "failed_folders": sorted(failed_folders),
        "skipped_folders": sorted(skipped_folders),
        "updated_at": _now_iso(),
    }

    path = project_root() / "cache" / f"array_{array_id}" / "progress.json"
    _write_json_atomic(path, payload)
    return path


def write_run_progress_summary(num_arrays: int | None = None) -> Path:
    num_arrays = get_num_arrays() if num_arrays is None else num_arrays
    root = project_root() / "cache"
    arrays: dict[str, Any] = {}
    total_assigned = 0
    total_completed = 0
    total_failed = 0
    total_skipped = 0

    total_remaining = 0

    for array_id in range(num_arrays):
        path = root / f"array_{array_id}" / "progress.json"
        data = _read_json(path)
        if data is None:
            continue
        key = str(array_id)
        arrays[key] = {
            "assigned": data.get("assigned", 0),
            "completed_count": data.get("completed_count", 0),
            "failed_count": data.get("failed_count", 0),
            "skipped_count": data.get("skipped_count", 0),
            "remaining_count": data.get("remaining_count", 0),
            "completed_folders": data.get("completed_folders", []),
            "failed_folders": data.get("failed_folders", []),
            "skipped_folders": data.get("skipped_folders", []),
            "updated_at": data.get("updated_at"),
        }
        total_assigned += arrays[key]["assigned"]
        total_completed += arrays[key]["completed_count"]
        total_failed += arrays[key]["failed_count"]
        total_skipped += arrays[key]["skipped_count"]
        total_remaining += arrays[key]["remaining_count"]

    payload: dict[str, Any] = {
        "num_arrays": num_arrays,
        "total_folders": total_assigned,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "total_remaining": total_remaining,
        "arrays": arrays,
        "updated_at": _now_iso(),
    }

    out = run_progress_path()
    lock_path = out.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _write_json_atomic(out, payload)
    return out
