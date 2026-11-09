import json
import re
import time
from pathlib import Path

import pandas as pd

from data_loader import Case
from metrics import evaluate, ZERO
from model_clients import call_model
from prompts import build_prompt, reveal_count, effective_version


def parse_response(text: str) -> pd.DataFrame:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    try:
        rows = json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("["), t.rfind("]")
        if a == -1 or b <= a:
            raise
        rows = json.loads(t[a:b + 1])
    if isinstance(rows, dict):
        lists = [v for v in rows.values() if isinstance(v, list)]
        if len(lists) == 1:
            rows = lists[0]
    if not isinstance(rows, list):
        raise ValueError("response JSON is not an array of rows")
    return pd.DataFrame(rows)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def run_root(results_dir, track: str, model_key: str) -> Path:
    d = Path(results_dir) / _safe(track) / _safe(model_key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_call(entry: dict, root: Path) -> None:
    with open(root / "calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_case(case: Case, model_key: str, track: str, root: Path) -> dict:
    version = effective_version(case.composition_type, track)
    stem = f"{_safe(case.case_id)}_{_safe(model_key)}_{version}"
    d = root / "raw_responses" / _safe(case.dataset)
    d.mkdir(parents=True, exist_ok=True)

    n_reveal = reveal_count(len(case.input_df), len(case.gold_df))
    prompt = build_prompt(case.input_df, case.gold_df, n_reveal,
                          has_orig_index=case.has_orig_index,
                          include_glossary=(version == "glossary"))
    (d / f"{stem}_PROMPT.txt").write_text(prompt, encoding="utf-8")

    base = {
        "case_id": case.case_id, "example_id": case.extra_metadata.get("example_id", "") or case.case_id,
        "dataset": case.dataset, "track": case.track, "model": model_key,
        "composition_type": case.composition_type, "run_track": track, "prompt_version": version,
        "has_orig_index": int(case.has_orig_index),
        "n_input_rows": len(case.input_df), "n_gold_rows": len(case.gold_df), "n_reveal_rows": n_reveal,
    }

    t0 = time.time()
    try:
        resp = call_model(model_key, prompt)
    except Exception as e:
        _log_call({**base, "error": str(e), "ts": time.strftime("%F %T")}, root)
        return {**base, "error": str(e)}
    latency = round(time.time() - t0, 2)

    (d / f"{stem}.txt").write_text(resp["text"], encoding="utf-8")
    (d / f"{stem}_API.json").write_text(json.dumps(resp.get("raw", {}), ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")

    parse_error, pred_df = "", None
    try:
        pred_df = parse_response(resp["text"])
        pred_df.to_csv(d / f"{stem}_PRED.csv", index=False)
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"
    if parse_error and resp.get("finish_reason") == "length":
        parse_error = "TRUNCATED (hit the output token cap): " + parse_error

    scores = evaluate(case.gold_df, pred_df) if pred_df is not None else dict(ZERO)
    row = {
        **base,
        **scores,
        "input_tokens": resp["input_tokens"],
        "output_tokens": resp["output_tokens"],
        "reasoning_tokens": resp.get("reasoning_tokens", 0),
        "latency_s": latency,
        "finish_reason": resp.get("finish_reason", ""),
        "parse_error": parse_error,
    }
    _log_call({**{k: row[k] for k in ("case_id", "dataset", "model", "input_tokens", "output_tokens",
                                       "reasoning_tokens", "latency_s", "finish_reason", "parse_error")},
               "ts": time.strftime("%F %T")}, root)
    return row
