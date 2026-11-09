from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .select_bucket_representatives import run_select_one_per_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label bucket members and pick wiki/idx reps per bucket "
            "for mixed / single_column (1) and structural (up to 5, "
            "distinct wikis) benchmarks"
        )
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        action="append",
        required=True,
        help="bucketing output dir (repeatable)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="root with wiki_* folders; enables skipping candidates whose input and output tables are identical",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = 0
    for raw in args.bucket_dir:
        bucket_dir = raw.resolve()
        if not bucket_dir.is_dir():
            print(f"Not a directory: {bucket_dir}", file=sys.stderr)
            continue
        try:
            meta = run_select_one_per_bucket(bucket_dir, root=args.root)
        except Exception as e:
            print(f"FAILED {bucket_dir}: {e}", file=sys.stderr)
            continue
        ok += 1
        print(f"Bucket dir:  {bucket_dir}")
        print(f"Buckets:     {meta['n_buckets_total']}")
        ls = meta["label_stats"]
        print("Step 1: tagged every pipeline in every bucket:")
        print(
            "  counts: "
            f"single={ls['n_single_column']}  multi={ls['n_multi_column']}  "
            f"structural={ls['n_structural']}  cell-level={ls['n_cell_level']}  "
            f"mixed={ls['n_mixed']}"
        )
        print(f"  assignments CSV updated: {meta['files']['assignments_with_tags']}")
        n_json = meta.get("n_bucket_json_updated")
        if n_json:
            print(f"  bucket JSON files updated: {n_json}")

        io_check = meta["identical_io_check"]
        if io_check["enabled"]:
            print(
                f"  identical-io check: ON (root={io_check['root']}), "
                f"checked {io_check['n_checked']}, skipped {io_check['n_skipped_total']}"
            )
        else:
            print("  identical-io check: OFF (no --root given)")

        print("Step 2: selected per eligible bucket:")
        for name, n in meta["n_selected"].items():
            bmeta = meta["benchmarks"][name]
            print(
                f"  {name:14s}  selected={n}  "
                f"buckets={bmeta['n_buckets_with_eligible']}  "
                f"per_bucket={bmeta['per_bucket']}  "
                f"skipped_identical_io={bmeta['n_skipped_identical_io']}"
            )
        print(f"Rule:        {meta['rule']}")
        print("Wrote:")
        for k, p in meta["files"].items():
            print(f"  {k}: {p}")
        print()
    return 0 if ok == len(args.bucket_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
