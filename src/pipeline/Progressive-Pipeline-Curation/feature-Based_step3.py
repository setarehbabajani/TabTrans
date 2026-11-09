from __future__ import annotations

import argparse
import importlib
import math
import sys
from collections import defaultdict
from pathlib import Path

from src.pipeline.output_store import PIPELINES_FILE

from .jsonl_bundle import (
    DEFAULT_ROOT,
    SELECTION_SUMMARY_FILE,
    load_active_indices,
    load_surviving_pipelines,
    selection_dir,
    wiki_folders,
    write_selection_bundle,
)

_features = importlib.import_module("src.pipeline.Progressive-Pipeline-Curation.feature-Based_step1")
_regions = importlib.import_module("src.pipeline.Progressive-Pipeline-Curation.feature-Based_step2")
compute_features = _features.compute_features
read_source_column_count = _features.read_source_column_count
build_regions = _regions.build_regions

REGION_MIN_PCT = 20

REGION_MAX_PCT = 100

UNCAPPED_REGION_FEATURES = frozenset({"dependency_type"})


def _is_uncapped_region(region: tuple) -> bool:
    return region[0] in UNCAPPED_REGION_FEATURES


def _region_bounds(
    count: int,
    *,
    min_pct: float,
    max_pct: float,
    uncapped: bool = False,
) -> tuple[int, int]:
    if count <= 0:
        return 0, 0
    min_required = max(1, math.ceil(min_pct / 100.0 * count))
    if uncapped:
        return min_required, count
    max_allowed = min(count, max(min_required, math.floor(max_pct / 100.0 * count)))
    return min_required, max_allowed


def _build_region_maps(
    outputs: list[dict],
) -> tuple[dict[tuple, set[int]], dict[int, set[tuple]], dict[int, dict]]:
    region_to_idxs: dict[tuple, set[int]] = defaultdict(set)
    idx_to_regions: dict[int, set[tuple]] = {}
    output_by_idx: dict[int, dict] = {}

    for output in outputs:
        idx = output["idx"]
        output_by_idx[idx] = output
        regions = set(output["regions"])
        idx_to_regions[idx] = regions
        for region in regions:
            region_to_idxs[region].add(idx)

    return region_to_idxs, idx_to_regions, output_by_idx


def _sort_key_tuple(features: dict, idx: int) -> tuple:
    return (
        features.get("cleanup_score", 0),
        features.get("root_coverage_ratio", 0),
        features.get("semantic_score", 0),
        -idx,
    )


def _pick_best_idx(
    *,
    needy: set[tuple],
    region_to_idxs: dict[tuple, set[int]],
    region_sorted: dict[tuple, list[int]],
    sort_key_by_idx: dict[int, tuple],
    idx_to_regions: dict[int, set[tuple]],
    remaining: set[int],
    blocked: set[int],
) -> int | None:
    needy_sizes = sorted({len(region_to_idxs[region]) for region in needy})
    for size in needy_sizes:
        level_best_idx: int | None = None
        level_best_sk: tuple | None = None
        for region in needy:
            if len(region_to_idxs[region]) != size:
                continue
            for idx in region_sorted[region]:
                if idx not in remaining or idx in blocked:
                    continue
                if not (idx_to_regions[idx] & needy):
                    continue
                sk = sort_key_by_idx[idx]
                if level_best_sk is None or sk > level_best_sk:
                    level_best_sk = sk
                    level_best_idx = idx
                break
        if level_best_idx is not None:
            return level_best_idx
    return None


def region_selection(
    outputs: list[dict],
    *,
    min_pct: float = REGION_MIN_PCT,
    max_pct: float = REGION_MAX_PCT,
    progress_every: int = 0,
    progress_fn=None,
) -> list[dict]:
    if not outputs:
        return []

    region_to_idxs, idx_to_regions, output_by_idx = _build_region_maps(outputs)

    min_required: dict[tuple, int] = {}
    max_allowed: dict[tuple, int] = {}
    for region, idxs in region_to_idxs.items():
        lo, hi = _region_bounds(
            len(idxs),
            min_pct=min_pct,
            max_pct=max_pct,
            uncapped=_is_uncapped_region(region),
        )
        min_required[region] = lo
        max_allowed[region] = hi

    selected: set[int] = set()
    region_count: dict[tuple, int] = {region: 0 for region in region_to_idxs}
    remaining = set(output_by_idx.keys())
    region_to_remaining = {
        region: set(idxs) for region, idxs in region_to_idxs.items()
    }
    sort_key_by_idx = {
        idx: _sort_key_tuple(output_by_idx[idx]["features"], idx)
        for idx in output_by_idx
    }
    region_sorted = {
        region: sorted(idxs, key=lambda i: sort_key_by_idx[i], reverse=True)
        for region, idxs in region_to_idxs.items()
    }
    blocked: set[int] = set()
    picks = 0

    def needy_regions() -> set[tuple]:
        return {
            region
            for region in region_to_idxs
            if region_count[region] < min_required[region]
        }

    while True:
        needy = needy_regions()
        if not needy:
            break

        best_idx = _pick_best_idx(
            needy=needy,
            region_to_idxs=region_to_idxs,
            region_sorted=region_sorted,
            sort_key_by_idx=sort_key_by_idx,
            idx_to_regions=idx_to_regions,
            remaining=remaining,
            blocked=blocked,
        )

        if best_idx is None:
            break

        selected.add(best_idx)
        remaining.remove(best_idx)
        blocked.add(best_idx)
        for region in idx_to_regions[best_idx]:
            region_to_remaining[region].discard(best_idx)
            region_count[region] += 1
            if region_count[region] >= max_allowed[region]:
                blocked.update(region_to_idxs[region])

        picks += 1
        if progress_every > 0 and progress_fn and picks % progress_every == 0:
            progress_fn(picks, len(needy))

    selected_outputs = [output_by_idx[idx] for idx in sorted(selected)]
    return selected_outputs


def process_wiki_selection(wiki_path):
    if not (wiki_path / PIPELINES_FILE).is_file():
        return None

    indices = sorted(load_active_indices(wiki_path))
    if not indices:
        return None

    source_csv = wiki_path / "source.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(f"Missing source.csv in {wiki_path}")
    total_source_columns = read_source_column_count(str(source_csv))

    pipelines = load_surviving_pipelines(wiki_path, indices)
    outputs = [
        {"idx": idx, "features": compute_features(pipelines[idx], total_source_columns)}
        for idx in indices
        if idx in pipelines
    ]
    del pipelines
    if not outputs:
        return None

    outputs = build_regions(outputs)
    selected = region_selection(outputs)
    selected_indices = [o["idx"] for o in selected]
    return write_selection_bundle(wiki_path, selected_indices, surviving_count=len(indices))


def main():
    parser = argparse.ArgumentParser(
        description="Feature-based selection: features, regions and region-coverage selection"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--wiki", help="process a single wiki folder")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip wikis that already have selection/selection_summary.json",
    )
    args = parser.parse_args()

    processed = skipped = 0
    total_selected = 0
    for wiki_path in wiki_folders(args.root, args.wiki):
        name = wiki_path.name
        if args.skip_existing and (selection_dir(wiki_path) / SELECTION_SUMMARY_FILE).is_file():
            print(f"{name}: skip (already done)", flush=True)
            continue
        try:
            summary = process_wiki_selection(wiki_path)
        except Exception as exc:
            print(f"ERROR {name}: {exc}", file=sys.stderr, flush=True)
            continue
        if summary is None:
            skipped += 1
            print(f"{name}: skip (nothing to select)", flush=True)
            continue
        processed += 1
        total_selected += summary["selected_count"]
        print(
            f"{name}: after_prune={summary['after_prune']} "
            f"-> selected={summary['selected_count']}",
            flush=True,
        )

    print(
        f"Selection finished: {processed} wiki(s), "
        f"{total_selected} outputs selected, {skipped} skipped",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
