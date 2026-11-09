import argparse
import csv
import sys
from pathlib import Path

from config import MODELS, RESULTS_DIR, DATA_DIR
from data_loader import LAYOUTS, list_cases, load_case
from metrics import KEYS
from prompts import TRACKS, effective_version
from runner import run_case, run_root

FIELDNAMES = [
    "case_id", "example_id", "dataset", "track", "model", "composition_type", "run_track", "prompt_version",
    "has_orig_index", "n_input_rows", "n_gold_rows", "n_reveal_rows",
    *KEYS,
    "input_tokens", "output_tokens", "reasoning_tokens", "latency_s",
    "finish_reason", "parse_error", "error", "status",
]


def existing_fieldnames(path: Path) -> list:
    if not path.exists() or path.stat().st_size == 0:
        return FIELDNAMES
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f), []) or FIELDNAMES


def already_succeeded(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("error"):
                done.add((r["case_id"], r["model"], (r.get("dataset") or "").strip(),
                          (r.get("prompt_version") or "").strip()))
    return done


def final_ids(path, value="1") -> set:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["example_id"] for r in csv.DictReader(f) if (r.get("final") or "").strip() == value}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=LAYOUTS, default="case")
    ap.add_argument("--data-dir", nargs="+", default=[str(DATA_DIR)])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--final-csv", default=None)
    ap.add_argument("--final-value", default="1")
    ap.add_argument("--track", choices=TRACKS, default=None)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    track = args.track
    if track is None:
        if sys.stdin.isatty():
            while track not in TRACKS:
                track = input(f"Track of this run? {'/'.join(TRACKS)} : ").strip().lower() or None
        else:
            track = "auto"
            print("no --track given and not interactive: using auto")
    print(f"track = {track}")

    for m in args.models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m}; choose from {list(MODELS)}")

    refs = []
    for dd in args.data_dir:
        found = list_cases(args.layout, dd, dataset=args.dataset if len(args.data_dir) == 1 else None)
        print(f"{dd}: {len(found)} case(s)")
        refs += found
    total_listed = len(refs)
    if args.cases:
        want = set(args.cases)
        refs = [r for r in refs if want & ({r.case_id} | r.keys)]
    if args.final_csv:
        wanted = final_ids(args.final_csv, args.final_value)
        refs = [r for r in refs if wanted & ({r.case_id} | r.keys)]
        print(f"{args.final_csv}: {len(wanted)} example(s) with final == {args.final_value}; "
              f"{len(refs)} of {total_listed} case(s) matched")
    if args.limit:
        refs = refs[: args.limit]
    if not refs:
        raise SystemExit(f"No cases to run (layout={args.layout}, data-dir={args.data_dir}).")

    skipped = 0
    for model_key in args.models:
        root = run_root(args.results_dir, track, model_key)
        results_file = root / "results_v2.csv"
        fieldnames = existing_fieldnames(results_file)
        done = set() if args.force else already_succeeded(results_file)
        print(f"\n=== {track} / {model_key} -> {results_file}  ({len(done)} already done)")
        write_header = (not results_file.exists()) or results_file.stat().st_size == 0
        with open(results_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for ref in refs:
                ver = effective_version(ref.composition_type, track)
                if ((ref.case_id, model_key, ref.dataset, ver) in done
                        or (ref.case_id, model_key, ref.dataset, "") in done):
                    skipped += 1
                    continue
                try:
                    case = load_case(ref)
                except Exception as e:
                    print(f"SKIPPING {ref.case_id}: failed to load ({e})")
                    writer.writerow({"case_id": ref.case_id, "dataset": ref.dataset, "model": model_key,
                                     "error": f"load failed: {e}"})
                    f.flush()
                    continue
                row = run_case(case, model_key, track, root)
                row["status"] = "error_row" if row.get("error") else "ok"
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                f.flush()
                print(f"done: {ref.dataset}/{ref.case_id}")
    if skipped:
        print(f"Skipped {skipped} already-completed pair(s).")


if __name__ == "__main__":
    main()
