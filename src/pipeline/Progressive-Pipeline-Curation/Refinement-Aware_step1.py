import argparse
import json
from pathlib import Path

from src.pipeline.output_store import PIPELINES_FILE

from .jsonl_bundle import (
    DEFAULT_ROOT,
    PRUNE_SUMMARY_FILE,
    SURVIVING_INDICES_FILE,
    CLEANUP_NAMES,
    load_active_indices,
    load_pipelines_by_idx,
    selection_dir,
    wiki_folders,
    write_indices,
)


def remove_raw_phase2_outputs(pipelines, active):
    outputs = []
    for idx in sorted(active):
        if idx not in pipelines:
            continue
        pipeline = pipelines[idx]
        phase3 = {s.get("transformation") for s in pipeline.get("phase3_structural", [])}
        outputs.append({
            "idx": idx,
            "phase2_id": pipeline.get("phase2_id"),
            "cleanup_only": bool(phase3) and phase3 <= CLEANUP_NAMES,
            "phase2_only": not phase3,
        })

    cleaned_ids = {
        o["phase2_id"] for o in outputs
        if o["phase2_id"] is not None and o["cleanup_only"]
    }
    return {
        o["idx"] for o in outputs
        if o["phase2_only"] and o["phase2_id"] in cleaned_ids
    }


def save_results(wiki_path, total, removed, surviving):
    sel_dir = selection_dir(wiki_path)
    write_indices(sel_dir / SURVIVING_INDICES_FILE, sorted(surviving))
    summary_path = sel_dir / PRUNE_SUMMARY_FILE
    summary = {}
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    summary.update({
        "wiki": wiki_path.name,
        "total_outputs": total,
        "step1_removed": removed,
        "surviving_count": len(surviving),
    })
    summary.setdefault("step2_removed", 0)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_step1(root, wiki=None):
    for wiki_path in wiki_folders(root, wiki):
        if not (wiki_path / PIPELINES_FILE).is_file():
            continue

        pipelines = load_pipelines_by_idx(wiki_path)
        active = load_active_indices(wiki_path)
        remove = remove_raw_phase2_outputs(pipelines, active)
        active -= remove
        save_results(wiki_path, len(pipelines), len(remove), active)

        print(
            f"{wiki_path.name}: total={len(pipelines)} - step1={len(remove)} "
            f"-> surviving={len(active)}"
        )

    print("Refinement-aware step 1 finished")


def main():
    parser = argparse.ArgumentParser(
        description="Refinement-aware step 1: remove raw phase-2 outputs that have a cleanup-only version"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--wiki", help="process a single wiki folder")
    args = parser.parse_args()
    run_step1(args.root, args.wiki)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
