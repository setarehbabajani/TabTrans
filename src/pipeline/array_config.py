from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_NUM_ARRAYS = 8


def project_root() -> Path:
    return Path(
        os.environ.get(
            "TABTRANS_ROOT",
            Path(__file__).resolve().parents[2],
        )
    )


def get_array_id() -> int:
    return int(os.environ.get("TABTRANS_ARRAY_ID", "0"))


def get_num_arrays() -> int:
    env = os.environ.get("TABTRANS_NUM_ARRAYS")
    if env is not None:
        return int(env)
    manifest = load_manifest()
    if manifest and manifest.get("num_arrays"):
        return int(manifest["num_arrays"])
    return DEFAULT_NUM_ARRAYS


def normalize_wiki_root(wiki_root: str) -> str:
    return str(Path(wiki_root).resolve())


def wiki_roots_equivalent(left: str, right: str) -> bool:
    if not left or not right:
        return False
    a = Path(left)
    b = Path(right)
    try:
        if a.exists() and b.exists():
            return a.resolve() == b.resolve()
    except OSError:
        pass
    return os.path.normpath(os.path.abspath(left)) == os.path.normpath(
        os.path.abspath(right)
    )


def array_dir() -> Path:
    path = project_root() / "cache" / f"array_{get_array_id()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tag_cache_path() -> Path:
    return array_dir() / "tag_cache.json"


def pipeline_errors_path() -> Path:
    return array_dir() / "pipeline_errors.jsonl"


def structural_check_cache_path() -> Path:
    return array_dir() / "structural_check_cache.json"


def manifest_path() -> Path:
    return project_root() / "cache" / "array_manifest.json"


def list_wiki_folders(wiki_root: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(wiki_root)
        if os.path.isdir(os.path.join(wiki_root, name))
    )


def shard_folders(
    folders: list[str],
    array_id: int,
    num_arrays: int,
) -> list[str]:
    folders = sorted(folders)
    return [
        name
        for index, name in enumerate(folders)
        if index % num_arrays == array_id
    ]


def build_manifest(wiki_root: str, num_arrays: int) -> dict[str, Any]:
    folders = list_wiki_folders(wiki_root)
    existing = load_manifest() or {}
    skipped = [
        name
        for name in (existing.get("skipped_folders") or [])
        if name in folders
    ]
    skipped_set = set(skipped)
    runnable = [name for name in folders if name not in skipped_set]
    arrays = {
        str(array_id): shard_folders(runnable, array_id, num_arrays)
        for array_id in range(num_arrays)
    }
    return {
        "num_arrays": num_arrays,
        "total_folders": len(runnable),
        "wiki_root": os.path.abspath(wiki_root),
        "arrays": arrays,
        "skipped_folders": sorted(skipped),
    }


def write_manifest(wiki_root: str, num_arrays: int) -> Path:
    manifest = build_manifest(wiki_root, num_arrays)
    return _save_manifest(manifest)


def _save_manifest(manifest: dict[str, Any]) -> Path:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def refresh_manifest(wiki_root: str, num_arrays: int) -> Path:
    existing = load_manifest()
    wiki_abs = normalize_wiki_root(wiki_root)
    present = set(list_wiki_folders(wiki_root))
    skipped = sorted(
        name
        for name in ((existing or {}).get("skipped_folders") or [])
        if name in present
    )
    skipped_set = set(skipped)

    if (
        existing is not None
        and existing.get("num_arrays") == num_arrays
        and isinstance(existing.get("arrays"), dict)
        and (
            wiki_roots_equivalent(str(existing.get("wiki_root", "")), wiki_root)
            or any(
                name in present
                for names in existing["arrays"].values()
                for name in names
            )
        )
    ):
        arrays = {
            str(array_id): [
                name
                for name in existing["arrays"].get(str(array_id), [])
                if name in present and name not in skipped_set
            ]
            for array_id in range(num_arrays)
        }
        manifest = {
            "num_arrays": num_arrays,
            "total_folders": sum(len(v) for v in arrays.values()),
            "wiki_root": wiki_abs,
            "arrays": arrays,
            "skipped_folders": skipped,
        }
    else:
        manifest = build_manifest(wiki_root, num_arrays)

    return _save_manifest(manifest)


def load_manifest() -> dict[str, Any] | None:
    path = manifest_path()
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Warning: corrupt manifest at {path}: {exc}")
        return None


def root_tag_cache_path() -> Path:
    return project_root() / "tag_cache.json"


def ensure_array_dirs(num_arrays: int) -> list[Path]:
    created = []
    for array_id in range(num_arrays):
        path = project_root() / "cache" / f"array_{array_id}"
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def seed_tag_caches(
    num_arrays: int,
    *,
    source: Path | str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    source_path = Path(source) if source is not None else root_tag_cache_path()
    if not source_path.is_file():
        raise FileNotFoundError(f"tag cache source not found: {source_path}")

    ensure_array_dirs(num_arrays)
    written: list[Path] = []
    for array_id in range(num_arrays):
        dest = project_root() / "cache" / f"array_{array_id}" / "tag_cache.json"
        if dest.exists() and not overwrite:
            continue
        shutil.copy2(source_path, dest)
        written.append(dest)
    return written


def setup_arrays(
    num_arrays: int,
    wiki_root: str,
    *,
    seed_tag_cache: bool = True,
    overwrite_tag_cache: bool = False,
) -> dict[str, Any]:
    dirs = ensure_array_dirs(num_arrays)
    manifest_path = refresh_manifest(wiki_root, num_arrays)
    seeded: list[str] = []
    if seed_tag_cache:
        seeded = [str(p) for p in seed_tag_caches(
            num_arrays,
            overwrite=overwrite_tag_cache,
        )]
    return {
        "num_arrays": num_arrays,
        "array_dirs": [str(d) for d in dirs],
        "manifest": str(manifest_path),
        "tag_caches_seeded": seeded,
    }


def manifest_folders_for_array(
    array_id: int,
    num_arrays: int,
) -> list[str] | None:
    manifest = load_manifest()
    if (
        manifest is None
        or manifest.get("num_arrays") != num_arrays
        or not isinstance(manifest.get("arrays"), dict)
    ):
        return None
    skipped = set(manifest.get("skipped_folders") or [])
    return [
        name
        for name in (manifest["arrays"].get(str(array_id), []) or [])
        if name not in skipped
    ]


def folders_for_array(
    wiki_root: str,
    array_id: int,
    num_arrays: int,
    *,
    use_manifest: bool = True,
) -> list[str]:
    if use_manifest:
        assigned = manifest_folders_for_array(array_id, num_arrays)
        if assigned is not None:
            present = set(list_wiki_folders(wiki_root))
            kept = [name for name in assigned if name in present]
            if kept or not assigned:
                return kept

    return shard_folders(list_wiki_folders(wiki_root), array_id, num_arrays)
