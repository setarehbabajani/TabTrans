from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .export_selected_sequences import DEFAULT_SAMPLE_SIZE, export_selected_sequences
from .jsonl_bundle import DEFAULT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export global selected pipeline sequence CSVs"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="root with wiki_* folders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <root>/_selection_analysis)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "number of pipelines to write full human-readable detail for, "
            f"in selected_pipeline_sequences_sample.csv (default: {DEFAULT_SAMPLE_SIZE})"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Root not found: {root}", file=sys.stderr)
        return 1

    out_dir = (args.out or (root / "_selection_analysis")).resolve()
    summary = export_selected_sequences(root, out_dir, sample_size=args.sample_size)

    print(f"Selected outputs:            {summary['n_selected_outputs']}")
    print(f"Wikis:                       {summary['n_wikis']}")
    print(f"Column patterns (anon seqs): {summary['n_column_patterns']}")
    print(
        "Avg / column pattern:        "
        f"{summary['avg_outputs_per_column_pattern']}"
    )
    print(
        f"Sampled detail rows:         {summary['n_sampled_detail_rows']} "
        f"(requested {summary['sample_size_requested']})"
    )
    print("\nTop column patterns:")
    for pat, n in summary["top_column_patterns"][:10]:
        short = pat if len(pat) <= 110 else pat[:107] + "..."
        print(f"  {n:6d}  {short}")
    print("\nWrote:")
    for key, path in summary["files"].items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
