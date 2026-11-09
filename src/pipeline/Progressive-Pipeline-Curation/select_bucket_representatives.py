from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from .exact_jaccard_column_patterns import parse_column_pattern_key
from .filter_identical_io import (
    load_multi_header,
    load_source_data_rows,
    check_identical_for_wiki,
)

ASSIGNMENT_CANDIDATES = (
    "subset_bucket_assignments.csv",
    "exact_bucket_assignments.csv",
    "lsh_bucket_assignments.csv",
)

LABEL_FIELDS = (
    "dependency_type",
    "composition_type",
    "n_phase2_seqs",
    "n_total_steps",
)

WRAP_STEP = "wrap_multi_header"

MULTI_COLUMN_TRANSFORMS: set[str] = set()

BENCHMARKS: tuple[tuple[str, str, Callable[["_Member"], bool], int], ...] = (
    ("mixed", "no filter (all members)", lambda m: True, 1),
    (
        "single_column",
        "dependency_type == single_column",
        lambda m: m.dependency_type == "single_column",
        1,
    ),
    (
        "structural",
        "composition_type == structural",
        lambda m: m.composition_type == "structural",
        5,
    ),
)


def _bare_name(tok: str) -> str:
    tok = (tok or "").strip()
    brace = tok.find("{")
    return tok[:brace] if brace != -1 else tok


def phase2_sequence_is_multi_column(names: list[str]) -> bool:
    awaiting_concat = False
    for n in names:
        if n == "split_column":
            awaiting_concat = True
        elif n == "concat_columns":
            if awaiting_concat:
                awaiting_concat = False
            else:
                return True
    return False


def pipeline_size_metrics(column_pattern_key: str) -> tuple[int, int]:
    parsed = parse_column_pattern_key(column_pattern_key, keep_params=False)
    phase1 = parsed["phase1"]
    phase2_columns = parsed["phase2_columns"]
    phase3 = parsed["phase3"]
    n_phase2_seqs = len(phase2_columns)
    n_total_steps = (
        len(phase1)
        + sum(len(col) for col in phase2_columns)
        + len(phase3)
    )
    return n_phase2_seqs, n_total_steps


def classify_pipeline(column_pattern_key: str) -> tuple[str, str]:
    parsed = parse_column_pattern_key(column_pattern_key, keep_params=False)
    phase1 = [_bare_name(t) for t in parsed["phase1"]]
    phase2_columns = [
        [_bare_name(t) for t in col] for col in parsed["phase2_columns"]
    ]
    phase3 = [_bare_name(t) for t in parsed["phase3"]]

    structural_names = phase1 + phase3

    has_non_wrap_structural = any(n != WRAP_STEP for n in structural_names)
    has_multi_phase2 = any(
        phase2_sequence_is_multi_column(col) for col in phase2_columns
    )

    all_step_names = set(phase1) | {n for col in phase2_columns for n in col} | set(phase3)
    has_multi_column_transform = bool(all_step_names & MULTI_COLUMN_TRANSFORMS)

    dependency_type = (
        "multi_column"
        if (has_non_wrap_structural or has_multi_phase2 or has_multi_column_transform)
        else "single_column"
    )

    has_cell_level = bool(phase2_columns)
    has_structural = bool(structural_names)
    if has_structural and has_cell_level:
        composition_type = "mixed"
    elif has_structural:
        composition_type = "structural"
    elif has_cell_level:
        composition_type = "cell-level"
    else:
        composition_type = "structural"
    return dependency_type, composition_type


@lru_cache(maxsize=None)
def _labels_for_key(column_pattern_key: str) -> tuple[str, str, int, int]:
    dep, comp = classify_pipeline(column_pattern_key)
    n_seqs, n_steps = pipeline_size_metrics(column_pattern_key)
    return dep, comp, n_seqs, n_steps


def _labels_for_row(column_pattern_key: str, n_cols_raw: Any) -> tuple[str, str, int, int]:
    dep, comp, n_seqs, n_steps = _labels_for_key(column_pattern_key)
    if n_cols_raw not in (None, ""):
        n_seqs = int(n_cols_raw)
    return dep, comp, n_seqs, n_steps


def find_assignments_csv(bucket_dir: Path) -> Path:
    for name in ASSIGNMENT_CANDIDATES:
        p = bucket_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No assignments CSV in {bucket_dir} "
        f"(looked for {ASSIGNMENT_CANDIDATES})"
    )


@dataclass
class _Member:
    bucket_id: int
    pattern_id: int
    wiki: str
    idx: int
    column_pattern_key: str
    n_phase2_seqs: int
    n_total_steps: int
    dependency_type: str
    composition_type: str


def load_members(assignments_csv: Path) -> dict[int, list[_Member]]:
    by_b: dict[int, list[_Member]] = defaultdict(list)
    with open(assignments_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"bucket_id", "wiki", "idx", "column_pattern_key"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{assignments_csv} must have columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        for i, raw in enumerate(reader):
            if (i + 1) % 1_000_000 == 0:
                print(f"  loaded {i + 1} bucket rows...", flush=True)
            key = raw["column_pattern_key"]
            dep, comp, n_seqs, n_steps = _labels_for_row(key, raw.get("n_cols"))
            by_b[int(raw["bucket_id"])].append(
                _Member(
                    bucket_id=int(raw["bucket_id"]),
                    pattern_id=int(raw.get("pattern_id") or 0),
                    wiki=raw.get("wiki") or "",
                    idx=int(raw.get("idx") or 0),
                    column_pattern_key=key,
                    n_phase2_seqs=n_seqs,
                    n_total_steps=n_steps,
                    dependency_type=dep,
                    composition_type=comp,
                )
            )
    return by_b


def _rank_key(m: _Member) -> tuple:
    return (-m.n_phase2_seqs, -m.n_total_steps, m.pattern_id)


def _row_for(bid: int, members: list[_Member], n_eligible: int, rank: int, m: _Member) -> dict[str, Any]:
    return {
        "bucket_id": bid,
        "bucket_size": len(members),
        "n_eligible": n_eligible,
        "rep_rank": rank,
        "wiki": m.wiki,
        "idx": m.idx,
        "pattern_id": m.pattern_id,
        "n_phase2_seqs": m.n_phase2_seqs,
        "n_total_steps": m.n_total_steps,
        "dependency_type": m.dependency_type,
        "composition_type": m.composition_type,
        "column_pattern_key": m.column_pattern_key,
    }


def select_all_benchmarks(
    by_b: dict[int, list[_Member]],
    benchmarks,
    root: Optional[Path],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], dict[str, int]]:
    states: dict[tuple[str, int], dict[str, Any]] = {}
    for name, _rule, pred, per_bucket in benchmarks:
        if per_bucket < 1:
            raise ValueError(f"per_bucket must be >= 1, got {per_bucket}")
        for bid in sorted(by_b.keys()):
            members = by_b[bid]
            candidates = [m for m in members if pred(m)]
            if not candidates:
                continue
            states[(name, bid)] = {
                "ranked": sorted(candidates, key=_rank_key),
                "pos": 0,
                "picks": [],
                "seen_wikis": set(),
                "per_bucket": per_bucket,
                "n_eligible": len(candidates),
            }

    identical: dict[tuple[str, int], bool] = {}
    warned_missing: set[str] = set()
    n_skipped = {name: 0 for name, *_ in benchmarks}
    n_checked = {name: 0 for name, *_ in benchmarks}

    def open_states():
        for key, st in states.items():
            if len(st["picks"]) < st["per_bucket"] and st["pos"] < len(st["ranked"]):
                yield key, st

    round_no = 0
    while True:
        pending = list(open_states())
        if not pending:
            break
        round_no += 1

        if root is not None:
            need: dict[str, set[int]] = defaultdict(set)
            for _key, st in pending:
                m = st["ranked"][st["pos"]]
                if m.wiki in st["seen_wikis"]:
                    continue
                if (m.wiki, m.idx) not in identical:
                    need[m.wiki].add(m.idx)
            n_wikis = len(need)
            print(
                f"  round {round_no}: {len(pending)} open (benchmark, bucket) slots, "
                f"checking {sum(len(v) for v in need.values())} candidates in {n_wikis} wikis...",
                flush=True,
            )
            for wi, (wiki, idxs) in enumerate(sorted(need.items()), 1):
                if wi % 1000 == 0:
                    print(f"    wiki {wi}/{n_wikis}...", flush=True)
                wiki_dir = root / wiki
                try:
                    multi_header = load_multi_header(wiki_dir)
                    input_rows = load_source_data_rows(wiki_dir, multi_header=multi_header)
                    res = check_identical_for_wiki(wiki_dir, idxs, input_rows)
                except FileNotFoundError as e:
                    if wiki not in warned_missing:
                        print(
                            f"[warn] {wiki}: {e} -- can't check identical-io, "
                            "keeping its candidates",
                            file=sys.stderr,
                        )
                        warned_missing.add(wiki)
                    res = {}
                for idx in idxs:
                    identical[(wiki, idx)] = bool(res.get(idx, False))

        for (name, bid), st in pending:
            m = st["ranked"][st["pos"]]
            st["pos"] += 1
            if m.wiki in st["seen_wikis"]:
                continue
            n_checked[name] += 1
            if root is not None and identical.get((m.wiki, m.idx), False):
                n_skipped[name] += 1
                continue
            st["picks"].append(m)
            st["seen_wikis"].add(m.wiki)

    selected: dict[str, list[dict[str, Any]]] = {name: [] for name, *_ in benchmarks}
    for (name, bid), st in sorted(states.items(), key=lambda kv: kv[0][1]):
        members = by_b[bid]
        for rank, m in enumerate(st["picks"], start=1):
            selected[name].append(_row_for(bid, members, st["n_eligible"], rank, m))
    if root is None:
        n_checked = {name: 0 for name in n_checked}
    return selected, n_skipped, n_checked


def _member_label_row(m: _Member) -> dict[str, Any]:
    return {
        "bucket_id": m.bucket_id,
        "pattern_id": m.pattern_id,
        "wiki": m.wiki,
        "idx": m.idx,
        "dependency_type": m.dependency_type,
        "composition_type": m.composition_type,
        "n_phase2_seqs": m.n_phase2_seqs,
        "n_total_steps": m.n_total_steps,
        "column_pattern_key": m.column_pattern_key,
    }


def write_labeled_members(
    out_path: Path,
    by_b: dict[int, list[_Member]],
) -> dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "bucket_id",
        "pattern_id",
        "wiki",
        "idx",
        "dependency_type",
        "composition_type",
        "n_phase2_seqs",
        "n_total_steps",
        "column_pattern_key",
    ]
    n = 0
    dep_counts: dict[str, int] = defaultdict(int)
    comp_counts: dict[str, int] = defaultdict(int)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for bid in sorted(by_b.keys()):
            for m in by_b[bid]:
                w.writerow(_member_label_row(m))
                n += 1
                dep_counts[m.dependency_type] += 1
                comp_counts[m.composition_type] += 1
    return {
        "n_labeled": n,
        "n_single_column": dep_counts.get("single_column", 0),
        "n_multi_column": dep_counts.get("multi_column", 0),
        "n_structural": comp_counts.get("structural", 0),
        "n_cell_level": comp_counts.get("cell-level", 0),
        "n_mixed": comp_counts.get("mixed", 0),
    }


def enrich_assignments_csv(
    assignments_csv: Path,
    by_b: dict[int, list[_Member]],
) -> Path:
    tmp = assignments_csv.with_suffix(assignments_csv.suffix + ".tmp")
    with open(assignments_csv, encoding="utf-8", newline="") as fin, \
            open(tmp, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        orig_fields = list(reader.fieldnames or [])
        base = [c for c in orig_fields if c not in LABEL_FIELDS]
        if "idx" in base:
            i = base.index("idx") + 1
            fields = base[:i] + list(LABEL_FIELDS) + base[i:]
        else:
            fields = base + list(LABEL_FIELDS)
        w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for raw in reader:
            dep, comp, n_seqs, n_steps = _labels_for_row(
                raw.get("column_pattern_key") or "", raw.get("n_cols")
            )
            out = dict(raw)
            out["dependency_type"] = dep
            out["composition_type"] = comp
            out["n_phase2_seqs"] = n_seqs
            out["n_total_steps"] = n_steps
            w.writerow(out)
    tmp.replace(assignments_csv)
    return assignments_csv


def enrich_buckets_json_dir(
    buckets_json_dir: Path,
    by_b: dict[int, list[_Member]],
) -> int | None:
    if not buckets_json_dir.is_dir():
        return None

    n_files = 0
    for path in sorted(buckets_json_dir.glob("bucket_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for p in payload.get("patterns") or []:
            dep, comp, n_seqs, n_steps = _labels_for_row(
                p.get("column_pattern_key") or "", p.get("n_cols")
            )
            p["dependency_type"] = dep
            p["composition_type"] = comp
            p["n_phase2_seqs"] = n_seqs
            p["n_total_steps"] = n_steps
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        n_files += 1
        if n_files % 50_000 == 0:
            print(f"  updated {n_files} bucket JSON files...", flush=True)
    return n_files


def write_selected(
    out_dir: Path,
    selected: list[dict[str, Any]],
    *,
    source_assignments: Path,
    bucket_dir: Path,
    benchmark: str,
    filter_rule: str,
    n_buckets_total: int,
    per_bucket: int = 1,
    n_skipped_identical_io: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "selected_pipelines.csv"
    json_path = out_dir / "selected_pipelines.json"
    wiki_idx_path = out_dir / "selected_wiki_idx.csv"
    meta_path = out_dir / "selection_meta.json"

    fields = [
        "bucket_id",
        "bucket_size",
        "n_eligible",
        "rep_rank",
        "wiki",
        "idx",
        "pattern_id",
        "n_phase2_seqs",
        "n_total_steps",
        "dependency_type",
        "composition_type",
        "column_pattern_key",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected)

    with open(wiki_idx_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["wiki", "idx", "bucket_id", "rep_rank"])
        w.writeheader()
        for row in selected:
            w.writerow(
                {
                    "wiki": row["wiki"],
                    "idx": row["idx"],
                    "bucket_id": row["bucket_id"],
                    "rep_rank": row.get("rep_rank", 1),
                }
            )

    json_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_buckets_with_eligible = len({row["bucket_id"] for row in selected})
    meta = {
        "benchmark": benchmark,
        "bucket_dir": str(bucket_dir),
        "source_assignments": str(source_assignments),
        "filter": filter_rule,
        "per_bucket": per_bucket,
        "n_buckets_total": n_buckets_total,
        "n_buckets_with_eligible": n_buckets_with_eligible,
        "n_selected": len(selected),
        "n_skipped_identical_io": n_skipped_identical_io,
        "rank_rule": (
            f"among eligible members: take top {per_bucket} by "
            "max n_phase2_seqs, then max n_total_steps, then lowest pattern_id "
            "(skipping identical input/output; distinct wiki/input table "
            "per pick within a bucket)"
        ),
        "files": {
            "selected_pipelines": str(csv_path),
            "selected_wiki_idx": str(wiki_idx_path),
            "selected_pipelines_json": str(json_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def run_select_one_per_bucket(
    bucket_dir: Path, root: Optional[Path] = None
) -> dict[str, Any]:
    bucket_dir = bucket_dir.resolve()
    assignments = find_assignments_csv(bucket_dir)
    print(f"Loading {assignments}...", flush=True)
    by_b = load_members(assignments)
    n_buckets_total = len(by_b)
    print(f"  {n_buckets_total} buckets; writing tags...", flush=True)

    labeled_path = bucket_dir / "labeled_bucket_pipelines.csv"
    label_stats = write_labeled_members(labeled_path, by_b)
    enrich_assignments_csv(assignments, by_b)
    n_json = enrich_buckets_json_dir(bucket_dir / "buckets_by_id", by_b)

    print("Selecting representatives...", flush=True)
    root_resolved = root.resolve() if root is not None else None
    selected_by, skipped_by, checked_by = select_all_benchmarks(
        by_b, BENCHMARKS, root_resolved
    )

    benchmarks_root = bucket_dir / "benchmarks"
    benchmark_metas: dict[str, Any] = {}
    for name, filter_rule, pred, per_bucket in BENCHMARKS:
        benchmark_metas[name] = write_selected(
            benchmarks_root / name,
            selected_by[name],
            source_assignments=assignments,
            bucket_dir=bucket_dir,
            benchmark=name,
            filter_rule=filter_rule,
            n_buckets_total=n_buckets_total,
            per_bucket=per_bucket,
            n_skipped_identical_io=skipped_by[name],
        )

    alias_meta = write_selected(
        bucket_dir / "selected_one_per_bucket",
        selected_by["mixed"],
        source_assignments=assignments,
        bucket_dir=bucket_dir,
        benchmark="mixed",
        filter_rule="no filter (all members); alias of benchmarks/mixed",
        n_buckets_total=n_buckets_total,
        per_bucket=1,
        n_skipped_identical_io=skipped_by["mixed"],
    )

    return {
        "bucket_dir": str(bucket_dir),
        "source_assignments": str(assignments),
        "n_buckets_total": n_buckets_total,
        "labeled_bucket_pipelines": str(labeled_path),
        "n_bucket_json_updated": n_json,
        "label_stats": label_stats,
        "benchmarks": benchmark_metas,
        "selected_one_per_bucket_alias": alias_meta,
        "identical_io_check": {
            "enabled": root_resolved is not None,
            "root": str(root_resolved) if root_resolved is not None else None,
            "n_checked": sum(checked_by.values()),
            "n_skipped_total": sum(skipped_by.values()),
        },
        "files": {
            "assignments_with_tags": str(assignments),
            "labeled_bucket_pipelines": str(labeled_path),
            "buckets_by_id": str(bucket_dir / "buckets_by_id"),
            **{
                f"benchmark_{name}": meta["files"]["selected_wiki_idx"]
                for name, meta in benchmark_metas.items()
            },
            "selected_one_per_bucket": alias_meta["files"]["selected_wiki_idx"],
        },
        "n_selected": {
            name: meta["n_selected"] for name, meta in benchmark_metas.items()
        },
        "rule": (
            "1) tag every pipeline in every bucket "
            "(composition_type: structural | cell-level | mixed); "
            "2) mixed/single_column: top 1 eligible per bucket; "
            "structural: top 5 eligible per bucket with distinct wikis "
            "(input tables); skip identical input/output candidates"
        ),
    }
