from __future__ import annotations

from typing import Any

WRAP = "wrap_multi_header"


def last_structural_step(steps, phase="phase1_structural") -> dict | None:
    for step in reversed(steps or []):
        if step.get("phase") != phase:
            continue
        if step.get("transformation") == WRAP:
            continue
        return step
    return None


def structural_depth(steps, phase="phase1_structural") -> int:
    return sum(
        1
        for s in (steps or [])
        if s.get("phase") == phase and s.get("transformation") != WRAP
    )


def columns_created_so_far(steps, *, phase="phase1_structural") -> set:
    created = set()
    for step in steps or []:
        if step.get("phase") != phase:
            continue
        for col in step.get("created") or []:
            created.add(str(col))
    return created


def blocks_transpose_repeat(steps, op_name, *, phase="phase1_structural"):
    if op_name != "transpose":
        return False, ""
    prev = last_structural_step(steps, phase=phase)
    if prev is None:
        return False, ""
    if prev.get("transformation") == "transpose":
        return True, "transpose -> transpose reproduces the input table"
    return False, ""


def _true_columns(column_flags: dict, kind: str) -> list:
    return [
        str(col)
        for col, info in (column_flags or {}).items()
        if isinstance(info, dict) and info.get(kind) is True
    ]


def plan_phase1_candidates(
    state,
    transforms_by_name: dict,
    flags: dict,
    *,
    max_depth: int | None = None,
    verbose: bool = True,
) -> list[tuple[Any, dict, list]]:
    out: list[tuple[Any, dict, list]] = []

    if max_depth is not None and structural_depth(state.steps) >= max_depth:
        if verbose:
            print(f"[PLAN] depth cap {max_depth} reached")
        return out

    df = state.df
    column_tags = [c.get("tag") for c in state.tags["columns"]]
    column_params = [c.get("parameters") for c in state.tags["columns"]]
    column_flags = flags.get("column_flags") or {}
    transpose_needed = flags.get("transpose_needed")

    def emit(op_name, kwargs, targets):
        t = transforms_by_name.get(op_name)
        if t is None:
            return
        blocked, reason = blocks_transpose_repeat(state.steps, op_name)
        if blocked:
            if verbose:
                print(f"[PLAN] skip {op_name}: {reason}")
            return
        out.append((t, kwargs, targets))

    created_before = columns_created_so_far(state.steps)

    approved_unpivot = [
        c for c in _true_columns(column_flags, "unpivot") if c in df.columns
    ]
    unpivot_cols = [c for c in approved_unpivot if c not in created_before]
    dropped = [c for c in approved_unpivot if c in created_before]
    if dropped and verbose:
        print(
            f"[PLAN] unpivot: ignoring {dropped} -- created by an earlier "
            "structural step, melting them would undo it"
        )
    if unpivot_cols:
        t = transforms_by_name.get("unpivot")
        if t is not None and t.check_for_columns(df, column_tags, unpivot_cols):
            emit("unpivot", {"value_cols": unpivot_cols}, unpivot_cols)
        elif verbose:
            print(f"[PLAN] unpivot on {unpivot_cols} cannot execute here")

    for col in _true_columns(column_flags, "pivot"):
        if col not in df.columns:
            continue
        t = transforms_by_name.get("pivot")
        if t is None:
            continue
        if col in created_before:
            if verbose:
                print(
                    f"[PLAN] skip pivot on {col!r} -- created by an earlier "
                    "structural step, pivoting it would undo that step"
                )
            continue
        if not t.check_for_column(
            df, column_tags, col, exclude_cols=created_before
        ):
            if verbose:
                print(f"[PLAN] pivot on {col!r} cannot execute here")
            continue
        emit(
            "pivot",
            {"column_col": col, "exclude_cols": list(created_before)},
            [col],
        )

    if transpose_needed is True:
        emit("transpose", {}, list(df.columns))
    elif verbose and transpose_needed is False:
        print("[PLAN] transpose.needed=false")

    t = transforms_by_name.get("divide_mixed")
    if t is not None:
        try:
            if t.check(df, column_tags, column_params):
                mixed_cols = [
                    str(c)
                    for c, tag in zip(df.columns, column_tags)
                    if tag == "COLUMN_MIXED"
                ]
                emit("divide_mixed", {}, mixed_cols)
            elif verbose:
                print("[PLAN] divide_mixed: no usable COLUMN_MIXED column")
        except Exception as exc:
            if verbose:
                print(f"[PLAN] divide_mixed check raised: {exc}")

    return out


def build_step_record(
    phase: str, transform, kwargs: dict, targets, created_diff=None
) -> dict:
    prov = getattr(transform, "last_provenance", None) or {}
    created = list(prov.get("created") or [])
    for col in created_diff or []:
        if str(col) not in {str(c) for c in created}:
            created.append(str(col))
    record = {
        "phase": phase,
        "transformation": transform.name,
        "columns": list(targets) if targets else None,
        "consumed": list(prov.get("consumed") or []),
        "created": created,
        "params": dict(kwargs) if kwargs else {},
    }
    protected = list(prov.get("protected") or [])
    if protected:
        record["protected"] = protected
    return record
