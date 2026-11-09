from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STOPWORD_TOKENS = {"P1:wrap_multi_header"}


_OPEN = "{[("
_CLOSE = "}])"


def _split_top_level(s: str, sep: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_string = False
    i, n, k = 0, len(s), len(sep)
    while i < n:
        c = s[i]
        if c == '"' and (i == 0 or s[i - 1] != "\\"):
            in_string = not in_string
            current.append(c)
            i += 1
        elif in_string:
            current.append(c)
            i += 1
        elif c in _OPEN:
            depth += 1
            current.append(c)
            i += 1
        elif c in _CLOSE:
            depth -= 1
            current.append(c)
            i += 1
        elif depth == 0 and s[i:i + k] == sep:
            parts.append("".join(current).strip())
            current = []
            i += k
        else:
            current.append(c)
            i += 1
    parts.append("".join(current).strip())
    return parts


def _parse_transform_token(tok: str, keep_params: bool) -> str:
    tok = tok.strip()
    brace = tok.find("{")
    if brace == -1:
        return tok
    name, param_str = tok[:brace], tok[brace:]
    if not keep_params:
        return name
    try:
        params = json.loads(param_str)
        return name + json.dumps(params, sort_keys=True)
    except json.JSONDecodeError:
        return tok


def parse_column_pattern_key(key: str, *, keep_params: bool) -> dict[str, Any]:
    def parse_flat_phase(s: str) -> list[str]:
        if not s or s.lower() == "none":
            return []
        return [t.strip() for t in _split_top_level(s, "|") if t.strip()]

    def parse_phase2(s: str) -> list[list[str]]:
        if not s or s.lower() == "none":
            return []
        columns: list[list[str]] = []
        for g in _split_top_level(s, "+"):
            g = g.strip()
            if g.startswith("[") and g.endswith("]"):
                g = g[1:-1].strip()
            seq = [_parse_transform_token(t, keep_params) for t in _split_top_level(g, ">") if t.strip()]
            columns.append(seq)
        return columns

    phases = _split_top_level((key or "").strip(), "||")
    if len(phases) != 3:
        return {"phase1": [], "phase2_columns": [], "phase3": []}

    phase1_str, phase2_str, phase3_str = [p.strip() for p in phases]
    return {
        "phase1": parse_flat_phase(phase1_str),
        "phase2_columns": parse_phase2(phase2_str),
        "phase3": parse_flat_phase(phase3_str),
    }


def tokens_from_parsed(parsed: dict[str, Any]) -> list[str]:
    tokens = [f"P1:{t}" for t in parsed["phase1"]]
    tokens = [t for t in tokens if t not in STOPWORD_TOKENS]

    col_seqs = sorted(parsed["phase2_columns"], key=lambda seq: "§".join(seq))
    for seq in col_seqs:
        tokens.append("COLSTART")
        tokens.extend(seq)
        tokens.append("COLEND")

    tokens.extend(f"P3:{t}" for t in parsed["phase3"])
    return tokens


def tokens_from_key(key: str, *, keep_params: bool) -> list[str]:
    return tokens_from_parsed(parse_column_pattern_key(key, keep_params=keep_params))


def ngrams(tokens: list[str], n: int) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) < n:
        return {"§".join(tokens)}
    return {"§".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def pattern_tokens(key: str, *, n: int = 3, keep_params: bool = True) -> frozenset[str]:
    toks = tokens_from_key(key, keep_params=keep_params)
    return frozenset(ngrams(toks, n))


@dataclass
class PatternRow:
    pattern_id: int
    column_pattern_key: str
    wiki: str
    idx: int
    shingles: frozenset[str]


def load_pattern_rows(
    path: Path,
    *,
    n: int = 3,
    limit: int | None = None,
    keep_params: bool = True,
) -> list[PatternRow]:
    rows: list[PatternRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"wiki", "idx", "column_pattern_key"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must have columns {sorted(required)}; got {reader.fieldnames}"
            )
        for i, raw in enumerate(reader):
            if limit is not None and i >= limit:
                break
            key = raw["column_pattern_key"]
            shingles = pattern_tokens(key, n=n, keep_params=keep_params)
            rows.append(
                PatternRow(
                    pattern_id=i,
                    column_pattern_key=key,
                    wiki=raw.get("wiki") or "",
                    idx=int(raw.get("idx") or 0),
                    shingles=shingles,
                )
            )
    return rows


_EPS = 1e-9


def _prefix_len(size: int, t: float) -> int:
    return size - math.ceil(t * size - _EPS) + 1


def build_exact_buckets(
    rows: list[PatternRow],
    *,
    threshold: float = 0.60,
    progress_every: int = 500_000,
    dedupe: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    t = threshold
    sets = [r.shingles for r in rows]

    freq: Counter = Counter()
    for s in sets:
        freq.update(s)
    tok_id = {tok: i for i, tok in enumerate(sorted(freq, key=lambda x: (freq[x], x)))}

    index: dict[int, list[int]] = defaultdict(list)
    leader_sets: list[frozenset[int]] = []
    leader_size: list[int] = []
    empty_leader: int | None = None
    cache: dict[frozenset[str], int] = {}
    bucket_of = [-1] * len(rows)
    n_rows = len(rows)
    n_verified = 0

    for i, s in enumerate(sets):
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  rows {i + 1}/{n_rows} (leaders={len(leader_sets)})...", flush=True)

        if dedupe and s in cache:
            bucket_of[i] = cache[s]
            continue

        if not s:
            if empty_leader is None:
                empty_leader = len(leader_sets)
                leader_sets.append(frozenset())
                leader_size.append(0)
            bid = empty_leader
        else:
            ids = sorted(tok_id[x] for x in s)
            a = len(ids)
            sset = frozenset(ids)
            lo, hi = t * a - _EPS, a / t + _EPS
            best_l, best_sim = -1, -1.0
            seen: set[int] = set()
            for tok in ids[:_prefix_len(a, t)]:
                for L in index.get(tok, ()):
                    if L in seen:
                        continue
                    seen.add(L)
                    b = leader_size[L]
                    if b < lo or b > hi:
                        continue
                    inter = len(sset & leader_sets[L])
                    sim = inter / (a + b - inter)
                    n_verified += 1
                    if sim > best_sim or (sim == best_sim and L < best_l):
                        best_sim, best_l = sim, L

            if best_l >= 0 and best_sim >= t:
                bid = best_l
            else:
                bid = len(leader_sets)
                leader_sets.append(sset)
                leader_size.append(a)
                for tok in ids[:_prefix_len(a, t)]:
                    index[tok].append(bid)

        bucket_of[i] = bid
        if dedupe:
            cache[s] = bid

    groups: dict[int, list[int]] = defaultdict(list)
    for i, bid in enumerate(bucket_of):
        groups[bid].append(i)

    ordered_bids = sorted(groups.keys(), key=lambda b: (-len(groups[b]), b))
    remap = {b: nb for nb, b in enumerate(ordered_bids)}
    dense = [remap[b] for b in bucket_of]

    sizes = [len(groups[b]) for b in ordered_bids]
    stats = {
        "n_patterns": len(rows),
        "n_buckets": len(ordered_bids),
        "n_leaders": len(leader_sets),
        "threshold": threshold,
        "method": "exact_jaccard_leader_prefix_filter_ngram_no_unigram",
        "dedupe": dedupe,
        "n_exact_jaccard_computed": n_verified,
        "singleton_buckets": sum(1 for s in sizes if s == 1),
        "multi_buckets": sum(1 for s in sizes if s > 1),
        "max_bucket_size": max(sizes) if sizes else 0,
        "avg_bucket_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "bucket_size_histogram_top": Counter(sizes).most_common(15),
    }
    print(
        f"  done. buckets={stats['n_buckets']} max_size={stats['max_bucket_size']} "
        f"exact_jaccards={n_verified}",
        flush=True,
    )
    return dense, stats


def pick_representatives(
    rows: list[PatternRow],
    bucket_of: list[int],
    *,
    per_bucket: int = 1,
) -> list[dict[str, Any]]:
    by_b: dict[int, list[PatternRow]] = defaultdict(list)
    for row, bid in zip(rows, bucket_of):
        by_b[bid].append(row)

    selected: list[dict[str, Any]] = []
    for bid in sorted(by_b.keys()):
        members = sorted(by_b[bid], key=lambda r: (r.pattern_id, r.wiki, r.idx))
        bucket_size = len(members)
        for rank, row in enumerate(members[:per_bucket]):
            selected.append(
                {
                    "bucket_id": bid,
                    "bucket_size": bucket_size,
                    "rep_rank": rank,
                    "pattern_id": row.pattern_id,
                    "wiki": row.wiki,
                    "idx": row.idx,
                    "column_pattern_key": row.column_pattern_key,
                }
            )
    return selected


def write_bucket_assignments(path: Path, rows: list[PatternRow], bucket_of: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["bucket_id", "pattern_id", "wiki", "idx", "column_pattern_key"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row, bid in zip(rows, bucket_of):
            w.writerow(
                {
                    "bucket_id": bid,
                    "pattern_id": row.pattern_id,
                    "wiki": row.wiki,
                    "idx": row.idx,
                    "column_pattern_key": row.column_pattern_key,
                }
            )


def write_representatives_csv(path: Path, reps: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not reps:
        path.write_text("", encoding="utf-8")
        return
    fields = list(reps[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(reps)


def write_bucket_summary_csv(path: Path, rows: list[PatternRow], bucket_of: list[int]) -> None:
    by_b: dict[int, list[PatternRow]] = defaultdict(list)
    for row, bid in zip(rows, bucket_of):
        by_b[bid].append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["bucket_id", "bucket_size", "n_wikis", "rep_wiki", "rep_idx", "rep_column_pattern_key"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for bid in sorted(by_b.keys()):
            members = by_b[bid]
            rep = min(members, key=lambda r: (r.pattern_id, r.wiki, r.idx))
            w.writerow(
                {
                    "bucket_id": bid,
                    "bucket_size": len(members),
                    "n_wikis": len({r.wiki for r in members}),
                    "rep_wiki": rep.wiki,
                    "rep_idx": rep.idx,
                    "rep_column_pattern_key": rep.column_pattern_key,
                }
            )


def _row_to_dict(row: PatternRow, bucket_id: int) -> dict[str, Any]:
    return {
        "bucket_id": bucket_id,
        "pattern_id": row.pattern_id,
        "wiki": row.wiki,
        "idx": row.idx,
        "column_pattern_key": row.column_pattern_key,
    }


def write_buckets_json_dir(out_dir: Path, rows: list[PatternRow], bucket_of: list[int]) -> Path:
    by_b: dict[int, list[PatternRow]] = defaultdict(list)
    for row, bid in zip(rows, bucket_of):
        by_b[bid].append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("bucket_*.json"):
        old.unlink()
    index: list[dict[str, Any]] = []
    for bid in sorted(by_b.keys()):
        members = sorted(by_b[bid], key=lambda r: (r.pattern_id, r.wiki, r.idx))
        payload = {
            "bucket_id": bid,
            "bucket_size": len(members),
            "patterns": [_row_to_dict(r, bid) for r in members],
        }
        fname = f"bucket_{bid:06d}.json"
        (out_dir / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append({"bucket_id": bid, "bucket_size": len(members), "file": fname})
    (out_dir / "index.json").write_text(
        json.dumps({"n_buckets": len(index), "n_patterns": len(rows), "buckets": index}, indent=2),
        encoding="utf-8",
    )
    return out_dir


def default_exact_out_dir(csv_path: Path, *, threshold: float, keep_params: bool, n: int = 3) -> Path:
    tag = "params" if keep_params else "noparams"
    thr = f"{threshold:.2f}".replace(".", "")
    return csv_path.parent / f"exact_buckets_{tag}_n{n}_t{thr}"


def run_exact_on_csv(
    csv_path: Path,
    out_dir: Path,
    *,
    threshold: float = 0.60,
    n: int = 3,
    per_bucket: int = 1,
    limit: int | None = None,
    keep_params: bool = False,
) -> dict[str, Any]:
    rows = load_pattern_rows(csv_path, n=n, limit=limit, keep_params=keep_params)
    bucket_of, stats = build_exact_buckets(rows, threshold=threshold)
    reps = pick_representatives(rows, bucket_of, per_bucket=per_bucket)

    out_dir.mkdir(parents=True, exist_ok=True)
    assign_path = out_dir / "exact_bucket_assignments.csv"
    rep_path = out_dir / "exact_bucket_representatives.csv"
    summary_path = out_dir / "exact_bucket_summary.csv"
    stats_path = out_dir / "exact_bucket_stats.json"
    buckets_json_dir = out_dir / "buckets_by_id"

    write_bucket_assignments(assign_path, rows, bucket_of)
    write_representatives_csv(rep_path, reps)
    write_bucket_summary_csv(summary_path, rows, bucket_of)
    write_buckets_json_dir(buckets_json_dir, rows, bucket_of)

    stats = {
        **stats,
        "n": n,
        "keep_params": keep_params,
        "per_bucket": per_bucket,
        "n_representatives": len(reps),
        "files": {
            "assignments": str(assign_path),
            "representatives": str(rep_path),
            "bucket_summary": str(summary_path),
            "stats": str(stats_path),
            "buckets_by_id": str(buckets_json_dir),
            "buckets_index": str(buckets_json_dir / "index.json"),
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
