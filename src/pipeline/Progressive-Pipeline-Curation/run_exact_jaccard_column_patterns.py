from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .exact_jaccard_column_patterns import default_exact_out_dir, run_exact_on_csv
from .jsonl_bundle import DEFAULT_ROOT


def parse_args() -> argparse.Namespace:
    default_csv = (
        Path(DEFAULT_ROOT).resolve()
        / "_selection_analysis"
        / "selected_pipeline_column_patterns.csv"
    )
    parser = argparse.ArgumentParser(
        description="Bucket column patterns by exact Jaccard similarity of n-gram shingles"
    )
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <csv_dir>/exact_buckets_params_n<N>_tXXX",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Jaccard threshold to join a bucket leader (default 0.60)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="n-gram size over the token sequence (default 3)",
    )
    parser.add_argument("--per-bucket", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    out_dir = (
        args.out
        or default_exact_out_dir(csv_path, threshold=args.threshold, keep_params=True, n=args.n)
    ).resolve()

    print(f"CSV:        {csv_path}")
    print(f"Out:        {out_dir}")
    print(f"Threshold:  {args.threshold}")
    print(f"n (n-gram): {args.n}")
    print(f"Per bucket: {args.per_bucket}")
    if args.limit:
        print(f"Limit:      {args.limit}")
    print("Loading + comparing...", flush=True)

    stats = run_exact_on_csv(
        csv_path,
        out_dir,
        threshold=args.threshold,
        n=args.n,
        per_bucket=args.per_bucket,
        limit=args.limit,
        keep_params=True,
    )

    print()
    print(f"Patterns:            {stats['n_patterns']}")
    print(f"Buckets:             {stats['n_buckets']}")
    print(f"Leaders:             {stats['n_leaders']}")
    print(f"Singleton buckets:   {stats['singleton_buckets']}")
    print(f"Multi-member buckets:{stats['multi_buckets']}")
    print(f"Max bucket size:     {stats['max_bucket_size']}")
    print(f"Avg bucket size:     {stats['avg_bucket_size']:.3f}")
    print(f"Representatives:     {stats['n_representatives']}")
    print("Bucket size histogram (size -> n_buckets):")
    for size, n in stats["bucket_size_histogram_top"]:
        print(f"  size={size:5d}  buckets={n}")
    print("\nWrote:")
    for k, p in stats["files"].items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
