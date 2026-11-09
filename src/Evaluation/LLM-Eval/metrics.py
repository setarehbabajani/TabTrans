import numpy as np
import pandas as pd

KEYS = ["sm_col_p", "sm_col_r", "sm_col_f1", "st_col_p", "st_col_r", "st_col_f1", "em"]
ZERO = {k: 0.0 for k in KEYS} | {"em": 0}


def _to_native(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(v, str) and v.strip().lower() in ("", "nan")


def _cell_equal(a, b) -> bool:
    a, b = _to_native(a), _to_native(b)
    a_missing, b_missing = _is_missing(a), _is_missing(b)
    if a_missing or b_missing:
        return a_missing and b_missing
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return abs(a - b) < 1e-9
    return a == b


def _columns(df: pd.DataFrame) -> dict:
    out = {}
    for i, name in enumerate(df.columns):
        if name not in out:
            out[name] = df.iloc[:, i].tolist()
    return out


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    gold, pred = _columns(gold_df), _columns(pred_df)
    n_gc, n_pc = len(gold), len(pred)

    sum_p = sum_r = 0.0
    n_right = 0
    for name, g in gold.items():
        p = pred.get(name)
        if p is None:
            continue
        correct = sum(_cell_equal(a, b) for a, b in zip(g, p))
        sum_p += correct / len(p) if p else 0.0
        sum_r += correct / len(g) if g else 1.0
        if len(p) == len(g) and correct == len(g):
            n_right += 1

    if n_gc == 0 and n_pc == 0:
        sm_p = sm_r = st_p = st_r = 1.0
    else:
        sm_p = sum_p / n_pc if n_pc else 0.0
        sm_r = sum_r / n_gc if n_gc else 1.0
        st_p = n_right / n_pc if n_pc else 0.0
        st_r = n_right / n_gc if n_gc else 1.0

    em = int(
        len(pred_df) == len(gold_df)
        and set(pred) == set(gold)
        and len(pred) == len(pred_df.columns)
        and all(
            len(pred[n]) == len(g) and all(_cell_equal(a, b) for a, b in zip(g, pred[n]))
            for n, g in gold.items()
        )
    )
    return {"sm_col_p": round(sm_p, 4), "sm_col_r": round(sm_r, 4), "sm_col_f1": round(f1(sm_p, sm_r), 4),
            "st_col_p": round(st_p, 4), "st_col_r": round(st_r, 4), "st_col_f1": round(f1(st_p, st_r), 4),
            "em": em}
