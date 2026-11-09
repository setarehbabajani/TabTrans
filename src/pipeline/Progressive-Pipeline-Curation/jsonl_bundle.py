from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.output_store import PIPELINES_FILE

DEFAULT_ROOT = Path("/path/to/dataset")

SELECTION_DIR = "selection"
SURVIVING_INDICES_FILE = "surviving_indices.json"
SURVIVING_PIPELINES_FILE = "surviving_pipelines.jsonl"
SELECTED_INDICES_FILE = "selected_indices.json"
PRUNE_SUMMARY_FILE = "prune_summary.json"
SELECTION_SUMMARY_FILE = "selection_summary.json"
SELECTED_PIPELINES_FILE = "selected_pipelines.jsonl"

CLEANUP_NAMES = {"drop_columns", "drop_duplicates", "drop_rows"}


def selection_dir(wiki_path):
    return Path(wiki_path) / SELECTION_DIR


def wiki_folders(root, wiki=None):
    root = Path(root)
    folders = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("wiki")
    )
    if wiki:
        folders = [p for p in folders if p.name == wiki]
    return folders


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_pipelines_by_idx(wiki_path, indices=None):
    wanted = set(indices) if indices is not None else None
    out = {}
    for record in read_jsonl(Path(wiki_path) / PIPELINES_FILE):
        idx = int(record["idx"])
        if wanted is None or idx in wanted:
            out[idx] = record["pipeline"]
    return out


def load_surviving_pipelines(wiki_path, indices):
    wanted = set(indices)
    if not wanted:
        return {}

    cache_path = selection_dir(wiki_path) / SURVIVING_PIPELINES_FILE
    if cache_path.is_file():
        cached = {int(r["idx"]): r["pipeline"] for r in read_jsonl(cache_path)}
        if wanted.issubset(cached.keys()):
            return {idx: cached[idx] for idx in indices}

    out = load_pipelines_by_idx(wiki_path, wanted)
    if out:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for idx in sorted(out):
                f.write(json.dumps({"idx": idx, "pipeline": out[idx]}, ensure_ascii=False) + "\n")
    return {idx: out[idx] for idx in indices if idx in out}


def write_indices(path, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(indices), f, indent=2)


def read_indices(path):
    with open(path, encoding="utf-8") as f:
        return sorted(int(i) for i in json.load(f))


def load_active_indices(wiki_path):
    path = selection_dir(wiki_path) / SURVIVING_INDICES_FILE
    if path.is_file():
        return set(read_indices(path))
    return set(load_pipelines_by_idx(wiki_path))


def write_selection_bundle(wiki_path, selected_indices, surviving_count):
    wiki_path = Path(wiki_path)
    sel_dir = selection_dir(wiki_path)
    keep = set(selected_indices)

    write_indices(sel_dir / SELECTED_INDICES_FILE, selected_indices)

    total = 0
    written = 0
    with open(sel_dir / SELECTED_PIPELINES_FILE, "w", encoding="utf-8") as out_f:
        for record in read_jsonl(wiki_path / PIPELINES_FILE):
            total += 1
            if int(record["idx"]) in keep:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    summary = {
        "wiki": wiki_path.name,
        "total_outputs": total,
        "after_prune": surviving_count,
        "selected_count": len(selected_indices),
        "selected_pipelines_lines": written,
    }
    with open(sel_dir / SELECTION_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
