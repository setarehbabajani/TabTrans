from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .jsonl_bundle import SELECTED_PIPELINES_FILE, read_jsonl, selection_dir
from .pipeline_signature import _roots_for_column, _step_key

DEFAULT_SAMPLE_SIZE = 40


def _step_name(step: dict[str, Any]) -> str:
    return str(step.get("transformation") or "")


def _phase_names(steps: list[dict[str, Any]] | None) -> str:
    names = [_step_name(s) for s in (steps or []) if _step_name(s)]
    return "|".join(names) if names else "none"


def _phase_names_with_columns(steps: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for step in steps or []:
        name = _step_name(step)
        if not name:
            continue
        cols = step.get("columns")
        if cols:
            col_txt = ",".join(str(c) for c in cols)
            parts.append(f"{name}[{col_txt}]")
        else:
            parts.append(name)
    return "|".join(parts) if parts else "none"


def _step_name_with_params(step: dict[str, Any]) -> str:
    name = _step_name(step)
    if not name:
        return ""
    params = step.get("params")
    if params is None or params == {}:
        return name
    param_txt = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}{param_txt}"


def _phase_names_with_params(steps: list[dict[str, Any]] | None) -> str:
    parts = [_step_name_with_params(s) for s in (steps or []) if _step_name(s)]
    return "|".join(parts) if parts else "none"


def _template_key(pipeline: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _phase_names(pipeline.get("phase1_structural")),
        _phase_names(pipeline.get("phase2_cell_level")),
        _phase_names(pipeline.get("phase3_structural")),
    )


def _template_with_params_key(pipeline: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _phase_names_with_params(pipeline.get("phase1_structural")),
        _phase_names_with_params(pipeline.get("phase2_cell_level")),
        _phase_names_with_params(pipeline.get("phase3_structural")),
    )


_PATTERN_KEY_NAME_ONLY = frozenset({"numeric_transformation"})


def _format_col_seq(seq: tuple[tuple[str, str], ...]) -> str:
    parts: list[str] = []
    for name, params_json in seq:
        if name in _PATTERN_KEY_NAME_ONLY or params_json in ("{}", "null", ""):
            parts.append(name)
        else:
            parts.append(f"{name}{params_json}")
    return " > ".join(parts) if parts else "none"


def _read_source_column_order(wiki_path: Path) -> list[str]:
    for candidate in (
        wiki_path / "source.csv",
        selection_dir(wiki_path) / "source.csv",
    ):
        if not candidate.is_file():
            continue
        with open(candidate, encoding="utf-8", newline="") as f:
            row = next(csv.reader(f), None)
        return [str(c) for c in (row or [])]
    return []


def column_pattern_key(
    pipeline: dict[str, Any],
    *,
    input_columns: list[str] | None = None,
) -> str:
    phase1 = _phase_names(pipeline.get("phase1_structural"))
    phase3 = _phase_names(pipeline.get("phase3_structural"))

    column_sequences: dict[str, list[tuple[str, str]]] = {}
    for step in pipeline.get("phase2_cell_level") or []:
        cols = step.get("columns") or []
        key = _step_key(step)
        step_roots: set[str] = set()
        for col in cols:
            step_roots.update(_roots_for_column(str(col)))
        for root in step_roots:
            column_sequences.setdefault(root, []).append(key)

    if not column_sequences:
        phase2 = "none"
    elif input_columns is not None:
        col_index = {c: i for i, c in enumerate(input_columns)}

        def _root_order(root: str) -> tuple:
            idx = col_index.get(root)
            if idx is None:
                return (1, root)
            return (0, idx)

        ordered_roots = sorted(column_sequences.keys(), key=_root_order)
        anon_seqs = [
            _format_col_seq(tuple(column_sequences[r])) for r in ordered_roots
        ]
        phase2 = " + ".join(f"[{s}]" for s in anon_seqs)
    else:
        anon_seqs = sorted(
            _format_col_seq(tuple(seq)) for seq in column_sequences.values()
        )
        phase2 = " + ".join(f"[{s}]" for s in anon_seqs)

    return " || ".join((phase1, phase2, phase3))


def iter_selected_wikis(root: Path) -> Iterator[Path]:
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not p.name.startswith("wiki"):
            continue
        if (selection_dir(p) / SELECTED_PIPELINES_FILE).is_file():
            yield p


def iter_selected_records(
    root: Path,
) -> Iterator[tuple[str, int, dict[str, Any], list[str]]]:
    for wiki_path in iter_selected_wikis(root):
        path = selection_dir(wiki_path) / SELECTED_PIPELINES_FILE
        input_columns = _read_source_column_order(wiki_path)
        for record in read_jsonl(path):
            idx = int(record["idx"])
            pipeline = record["pipeline"]
            yield wiki_path.name, idx, pipeline, input_columns


SAMPLE_FIELDS = [
    "wiki",
    "idx",
    "phase1_seq",
    "phase2_seq",
    "phase3_seq",
    "full_seq",
    "phase1_detail",
    "phase2_detail",
    "phase3_detail",
    "full_detail",
    "template_key",
    "template_with_params_key",
    "column_pattern_key",
    "n_phase1",
    "n_phase2",
    "n_phase3",
    "n_phase2_columns",
    "total_depth",
]


def _sample_row(
    wiki: str, idx: int, pipeline: dict[str, Any], cpattern: str
) -> dict[str, Any]:
    p1 = pipeline.get("phase1_structural") or []
    p2 = pipeline.get("phase2_cell_level") or []
    p3 = pipeline.get("phase3_structural") or []

    phase1_seq = _phase_names(p1)
    phase2_seq = _phase_names(p2)
    phase3_seq = _phase_names(p3)
    parts = [s for s in (phase1_seq, phase2_seq, phase3_seq) if s != "none"]
    full_seq = " || ".join(parts) if parts else "none"

    phase1_detail = _phase_names_with_columns(p1)
    phase2_detail = _phase_names_with_columns(p2)
    phase3_detail = _phase_names_with_columns(p3)
    detail_parts = [
        s for s in (phase1_detail, phase2_detail, phase3_detail) if s != "none"
    ]
    full_detail = " || ".join(detail_parts) if detail_parts else "none"

    tkey = _template_key(pipeline)
    tkey_params = _template_with_params_key(pipeline)

    roots: set[str] = set()
    for step in p2:
        for col in step.get("columns") or []:
            roots.update(_roots_for_column(str(col)))

    return {
        "wiki": wiki,
        "idx": idx,
        "phase1_seq": phase1_seq,
        "phase2_seq": phase2_seq,
        "phase3_seq": phase3_seq,
        "full_seq": full_seq,
        "phase1_detail": phase1_detail,
        "phase2_detail": phase2_detail,
        "phase3_detail": phase3_detail,
        "full_detail": full_detail,
        "template_key": " || ".join(tkey),
        "template_with_params_key": " || ".join(tkey_params),
        "column_pattern_key": cpattern,
        "n_phase1": len(p1),
        "n_phase2": len(p2),
        "n_phase3": len(p3),
        "n_phase2_columns": len(roots),
        "total_depth": len(p1) + len(p2) + len(p3),
    }


def export_selected_sequences(
    root: Path,
    out_dir: Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern_path = out_dir / "selected_pipeline_column_patterns.csv"
    sample_path = out_dir / "selected_pipeline_sequences_sample.csv"
    summary_path = out_dir / "selected_sequences_summary.json"

    n_selected = 0
    n_sampled = 0
    wikis: set[str] = set()
    pattern_counter: Counter[str] = Counter()

    with open(pattern_path, "w", encoding="utf-8", newline="") as pf, open(
        sample_path, "w", encoding="utf-8", newline=""
    ) as sf:
        pwriter = csv.DictWriter(pf, fieldnames=["wiki", "idx", "column_pattern_key"])
        pwriter.writeheader()
        swriter = csv.DictWriter(sf, fieldnames=SAMPLE_FIELDS)
        swriter.writeheader()

        for wiki, idx, pipeline, input_columns in iter_selected_records(root):
            n_selected += 1
            wikis.add(wiki)

            cpattern = column_pattern_key(pipeline, input_columns=input_columns)
            pattern_counter[cpattern] += 1
            pwriter.writerow(
                {"wiki": wiki, "idx": idx, "column_pattern_key": cpattern}
            )

            if n_sampled < sample_size:
                swriter.writerow(_sample_row(wiki, idx, pipeline, cpattern))
                n_sampled += 1

    n_patterns = len(pattern_counter)
    summary = {
        "n_selected_outputs": n_selected,
        "n_wikis": len(wikis),
        "n_column_patterns": n_patterns,
        "avg_outputs_per_column_pattern": (
            round(n_selected / n_patterns, 3) if n_patterns else 0
        ),
        "top_column_patterns": pattern_counter.most_common(10),
        "n_sampled_detail_rows": n_sampled,
        "sample_size_requested": sample_size,
        "files": {
            "pipeline_column_patterns": str(pattern_path),
            "pipeline_sequences_sample": str(sample_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["files"]["summary"] = str(summary_path)

    return summary
