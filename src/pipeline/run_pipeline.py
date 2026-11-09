import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from src.pipeline.array_config import (
    array_dir,
    folders_for_array,
    get_array_id,
    manifest_folders_for_array,
    pipeline_errors_path,
    project_root,
    setup_arrays,
    structural_check_cache_path,
    tag_cache_path,
    write_manifest,
)
from src.pipeline.array_progress import (
    folder_has_jsonl,
    scan_folders_with_jsonl,
    write_array_progress,
    write_run_progress_summary,
)
from src.tagging.call_llm import WikiTaggingFailed

INPUT_DIR = str(project_root() / "data" / "Datasets" / "wiki_tables")


def _configure_array(array_id: int, num_arrays: int) -> None:
    os.environ["TABTRANS_ARRAY_ID"] = str(array_id)
    os.environ["TABTRANS_NUM_ARRAYS"] = str(num_arrays)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def _attach_array_log(array_id: int):
    log_path = project_root() / "cache" / f"array_{array_id}" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    orig_out = sys.stdout
    orig_err = sys.stderr
    orig_name = getattr(orig_out, "name", None)
    if orig_name and os.path.abspath(orig_name) == os.path.abspath(log_path):
        return log_path
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(orig_out, log_file)
    sys.stderr = _Tee(orig_err, log_file)
    print(f"Logging to {log_path}")
    return log_path


def append_pipeline_error(wiki_folder, exc, stage="pipeline"):
    path = pipeline_errors_path()
    record = {
        "at": datetime.now(timezone.utc).isoformat(),
        "array_id": get_array_id(),
        "wiki_folder": wiki_folder,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run TabTrans pipeline on a deterministic wiki-folder array"
    )
    parser.add_argument(
        "--array-id",
        type=int,
        default=0,
        help="which array to run (0 .. num-arrays-1)",
    )
    parser.add_argument(
        "--num-arrays",
        type=int,
        default=8,
        help="total number of parallel arrays (default: 8)",
    )
    parser.add_argument(
        "--input-dir",
        default=INPUT_DIR,
        help="wiki_tables root directory",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write cache/array_manifest.json and exit",
    )
    parser.add_argument(
        "--setup-arrays",
        action="store_true",
        help="create cache/array_* dirs, manifest, and seed tag_cache.json",
    )
    parser.add_argument(
        "--overwrite-tag-cache",
        action="store_true",
        help="with --setup-arrays, replace existing per-array tag caches",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "skip wiki folders that already have any output jsonl "
            "(outputs/pipelines/output_tags), covers done and in-progress runs"
        ),
    )
    parser.add_argument(
        "--no-io",
        action="store_true",
        help="benchmark mode: skip writing output jsonl bundles only",
    )
    parser.add_argument(
        "--wiki",
        default=None,
        help="run only this wiki folder (must belong to --array-id)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.array_id < 0 or args.array_id >= args.num_arrays:
        print(
            f"array-id must be between 0 and {args.num_arrays - 1}",
            file=sys.stderr,
        )
        sys.exit(1)

    _configure_array(args.array_id, args.num_arrays)

    from src.pipeline.pipeline_engine import PipelineEngine

    if args.write_manifest:
        path = write_manifest(args.input_dir, args.num_arrays)
        print(f"Wrote manifest: {path}")
        return

    if args.setup_arrays:
        result = setup_arrays(
            args.num_arrays,
            args.input_dir,
            seed_tag_cache=True,
            overwrite_tag_cache=args.overwrite_tag_cache,
        )
        print(f"Created {len(result['array_dirs'])} array directories under cache/")
        print(f"Wrote manifest: {result['manifest']}")
        if result["tag_caches_seeded"]:
            print(f"Seeded {len(result['tag_caches_seeded'])} tag_cache.json files")
        else:
            print("Tag caches already present (use --overwrite-tag-cache to re-copy)")
        return

    _attach_array_log(args.array_id)
    array_dir()
    progress_assigned = manifest_folders_for_array(args.array_id, args.num_arrays)
    assigned_folders = folders_for_array(
        args.input_dir,
        args.array_id,
        args.num_arrays,
    )
    if progress_assigned is None:
        progress_assigned = list(assigned_folders)
    folders = list(assigned_folders)

    if args.wiki:
        wiki = args.wiki
        folder_path = os.path.join(args.input_dir, wiki)
        if not os.path.isdir(folder_path):
            print(f"wiki folder not found: {folder_path}", file=sys.stderr)
            sys.exit(1)
        if wiki not in progress_assigned:
            print(
                f"{wiki} is not assigned to array {args.array_id} "
                f"(num-arrays={args.num_arrays}). "
                "Use the array-id from cache/array_manifest.json.",
                file=sys.stderr,
            )
            sys.exit(1)
        tags_path = os.path.join(folder_path, "source_tags.json")
        if not os.path.isfile(tags_path):
            print(
                f"{wiki} has no source_tags.json; tag it first.",
                file=sys.stderr,
            )
            sys.exit(1)
        folders = [wiki]

    cache_dir = array_dir()
    print(
        f"\n=== Array {args.array_id + 1}/{args.num_arrays} "
        f"({len(folders)} wiki folders) ==="
    )
    print(f"Array cache dir: {cache_dir}")
    print(f"  tag_cache: {tag_cache_path()}")
    print(f"  structural:{structural_check_cache_path()}")
    print(f"  progress:  {cache_dir / 'progress.json'}")
    print(f"  errors:    {cache_dir / 'pipeline_errors.jsonl'}")

    completed_folders: list[str] = []
    failed_folders: list[str] = []
    skipped_folders: list[str] = []

    saved_progress_path = (
        project_root() / "cache" / f"array_{args.array_id}" / "progress.json"
    )
    if saved_progress_path.is_file():
        with open(saved_progress_path, encoding="utf-8") as f:
            saved_progress = json.load(f)
        for name in saved_progress.get("completed_folders", []):
            if name not in completed_folders:
                completed_folders.append(name)
        for name in saved_progress.get("failed_folders", []):
            if name not in failed_folders:
                failed_folders.append(name)
        for name in saved_progress.get("skipped_folders", []):
            if name not in skipped_folders:
                skipped_folders.append(name)
        if skipped_folders:
            print(
                f"{len(skipped_folders)} folders deferred/skipped "
                f"(from progress.json)"
            )

    if args.skip_existing:
        already_have_jsonl = scan_folders_with_jsonl(args.input_dir, folders)
        for name in already_have_jsonl:
            if name not in skipped_folders:
                skipped_folders.append(name)
        if already_have_jsonl:
            print(
                f"{len(already_have_jsonl)} folders already have jsonl "
                f"(done or in progress; will skip)"
            )

    def _save_progress() -> None:
        write_array_progress(
            wiki_root=args.input_dir,
            assigned_folders=progress_assigned,
            completed_folders=completed_folders,
            failed_folders=failed_folders,
            skipped_folders=skipped_folders,
            array_id=args.array_id,
            num_arrays=args.num_arrays,
        )
        write_run_progress_summary(args.num_arrays)

    _save_progress()

    if args.no_io:
        print(
            "NO-IO benchmark mode: skip outputs.jsonl / pipelines.jsonl / "
            "output_tags.jsonl only; tag cache + llm usage still saved"
        )

    engine = PipelineEngine(skip_io=args.no_io)

    for folder in folders:
        folder_path = os.path.join(args.input_dir, folder)

        if folder in skipped_folders:
            print(f"Skipping (deferred): {folder}")
            _save_progress()
            continue

        if folder in failed_folders:
            print(f"Skipping (previously failed): {folder}")
            _save_progress()
            continue

        if folder in completed_folders:
            print(f"Skipping (already completed): {folder}")
            _save_progress()
            continue

        if args.skip_existing and folder_has_jsonl(args.input_dir, folder):
            if folder not in skipped_folders:
                skipped_folders.append(folder)
            print(f"Skipping (jsonl present; done or in progress): {folder}")
            _save_progress()
            continue

        print(f"\nRunning pipeline on: {folder}")
        wiki_t0 = time.perf_counter()

        try:
            output_count = engine.run_on_folder(folder_path)
            wiki_elapsed = time.perf_counter() - wiki_t0
            print(
                f"Wiki wall time: {folder} - {wiki_elapsed:.1f}s "
                f"({output_count} outputs)"
            )

            if args.no_io:
                print(
                    f"Benchmark done (output jsonl skipped): "
                    f"{output_count} outputs"
                )
            else:
                completed_folders.append(folder)
                print(f"Done: {output_count} outputs generated")

            _save_progress()

        except WikiTaggingFailed as exc:
            failed_folders.append(folder)
            wiki_elapsed = time.perf_counter() - wiki_t0
            print(f"TAGGING/API FAILED in {folder} after {wiki_elapsed:.1f}s: {exc}")
            traceback.print_exc()
            append_pipeline_error(folder, exc, stage="tagging")
            _save_progress()
            print(
                f"ABORTING array {args.array_id} after tagging/API failure on "
                f"{folder}; not continuing to remaining wikis."
            )
            print(
                f"\n=== Array {args.array_id} aborted: "
                f"{len(completed_folders)} done, {len(failed_folders)} failed, "
                f"{len(skipped_folders)} skipped ==="
            )
            sys.exit(1)

        except Exception as exc:
            failed_folders.append(folder)
            wiki_elapsed = time.perf_counter() - wiki_t0
            print(f"Error in {folder} after {wiki_elapsed:.1f}s: {exc}")
            traceback.print_exc()

            append_pipeline_error(folder, exc)

            _save_progress()
            print(
                f"ABORTING array {args.array_id} after failure on {folder}; "
                f"not continuing to remaining wikis."
            )
            print(
                f"\n=== Array {args.array_id} aborted: "
                f"{len(completed_folders)} done, {len(failed_folders)} failed, "
                f"{len(skipped_folders)} skipped ==="
            )
            sys.exit(1)

    print(
        f"\n=== Array {args.array_id} finished: "
        f"{len(completed_folders)} done, {len(failed_folders)} failed, "
        f"{len(skipped_folders)} skipped ==="
    )
    _save_progress()


if __name__ == "__main__":
    main()
