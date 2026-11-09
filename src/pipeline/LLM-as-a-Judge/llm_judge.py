import argparse
import csv
import os
import re
import signal
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

HERE = Path(__file__).resolve().parent
with open(HERE / "prompt.txt", encoding="utf-8") as f:
    JUDGE_TEMPLATE = f.read()

FIELDNAMES = ["example_id", "track", "blind_score", "blind_reason"]


def env_int(name, default):
    try:
        value = int(os.environ.get(name, "") or default)
        return value if value > 0 else default
    except ValueError:
        return default


TABLE_MAX_ROWS = env_int("JUDGE_TABLE_MAX_ROWS", 60)
TABLE_MAX_CHARS = env_int("JUDGE_TABLE_MAX_CHARS", 20000)
CONTEXT_LIMIT = env_int("JUDGE_CONTEXT_LIMIT", 16384)
JUDGE_SEED = int(os.environ.get("JUDGE_SEED", "42"))
CALL_TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "300"))
ENABLE_THINKING = os.environ.get("JUDGE_ENABLE_THINKING", "1").strip().lower() not in ("0", "false", "no", "off")
THINKING_BUDGET = env_int("JUDGE_THINKING_BUDGET", 0)

JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "qwen3").strip().lower()

BACKENDS = {
    "qwen3": {
        "model": "qwen3-32b",
        "server_file": "_vllm_llmeval_qwen3.txt",
        "url_env": "JUDGE_BASE_URL",
        "key_env": "QWEN_SERVER_API_KEY",
    },
    "gpt_5_6_luna": {
        "model": "gpt-5.6-luna",
    },
}

if JUDGE_BACKEND not in BACKENDS:
    raise ValueError(f"Unknown JUDGE_BACKEND {JUDGE_BACKEND!r}. Use one of: {', '.join(BACKENDS)}")

BACKEND = BACKENDS[JUDGE_BACKEND]
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "").strip() or BACKEND["model"]


def extra_body():
    if JUDGE_BACKEND == "qwen3":
        body = {"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}}
        if ENABLE_THINKING and THINKING_BUDGET:
            body["thinking_token_budget"] = THINKING_BUDGET
        return body
    return {"reasoning_effort": "low"}


def project_root():
    env = os.environ.get("TABTRANS_ROOT", "").strip()
    if env:
        return Path(env)
    cluster = Path("/path/to/tabtrans")
    if cluster.is_dir():
        return cluster
    return HERE.parents[2]


def server_url():
    url = os.environ.get(BACKEND["url_env"], "").strip()
    if url:
        return url.rstrip("/")
    server_file = project_root() / BACKEND["server_file"]
    if server_file.is_file():
        hostport = server_file.read_text(encoding="utf-8").strip()
        if hostport.startswith(("http://", "https://")):
            return hostport.rstrip("/")
        if hostport:
            return f"http://{hostport}/v1"
    raise RuntimeError(
        f"No judge server for {JUDGE_BACKEND}: start the vLLM job that writes "
        f"{server_file}, or set {BACKEND['url_env']}=http://<host>:8000/v1"
    )


def make_client():
    if JUDGE_BACKEND == "gpt_5_6_luna":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=CALL_TIMEOUT)
    return OpenAI(
        base_url=server_url(),
        api_key=os.environ.get(BACKEND["key_env"], "not-needed"),
        max_retries=0,
        timeout=CALL_TIMEOUT,
    )


def table_markdown(csv_path, max_rows=TABLE_MAX_ROWS, max_chars=TABLE_MAX_CHARS,
                   min_rows=3, max_cell_chars=200, min_cell_chars=8):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    total_rows, total_cols = len(df), len(df.columns)
    df = df.head(max_rows)

    def render(n_rows, cell_cap=None):
        sub = df.head(n_rows)
        if cell_cap is not None:
            sub = sub.apply(lambda col: col.astype(str).apply(
                lambda v: v if len(v) <= cell_cap else v[:cell_cap] + "..."
            ))
        return sub.to_markdown(index=False)

    n_rows = len(df)
    md = render(n_rows)
    rows_reduced = n_rows < total_rows
    while len(md) > max_chars and n_rows > min_rows:
        n_rows = max(min_rows, n_rows // 2)
        md = render(n_rows)
        rows_reduced = True

    cells_shortened = False
    if len(md) > max_chars:
        cell_cap = max_cell_chars
        while True:
            md = render(n_rows, cell_cap)
            if len(md) <= max_chars or cell_cap == min_cell_chars:
                break
            cell_cap = max(min_cell_chars, cell_cap // 2)
        cells_shortened = True

    notes = []
    if rows_reduced:
        notes.append(f"showing {n_rows} of {total_rows} rows (all {total_cols} columns retained)")
    if cells_shortened:
        notes.append("some cell values shortened for length")
    if notes:
        md += "\n\n[... " + "; ".join(notes) + " ...]"
    return md


def call_llm(client, prompt, max_tokens=4000):
    available = CONTEXT_LIMIT - len(prompt) // 4 - 200
    max_tokens = max(256, min(max_tokens, available))

    if JUDGE_BACKEND == "gpt_5_6_luna":
        kwargs = {"max_completion_tokens": max_tokens}
    else:
        kwargs = {"max_tokens": max_tokens, "temperature": 0}

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        seed=JUDGE_SEED,
        messages=[{"role": "user", "content": prompt}],
        extra_body=extra_body(),
        **kwargs,
    )
    msg = response.choices[0].message
    return msg.content or getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""


def parse_reply(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    score = re.search(r"Score:\s*([01])", cleaned)
    reason = re.search(r"Reason:\s*(.+)", cleaned)
    return (int(score.group(1)) if score else None,
            reason.group(1).strip() if reason else None)


class SkipExample(Exception):
    pass


def judge_one(client, row, data_dir, max_cols, wide_cols, wide_rows):
    input_path = os.path.join(data_dir, row["input_csv"])
    output_path = os.path.join(data_dir, row["output_csv"])

    n_in = len(pd.read_csv(input_path, nrows=0).columns)
    n_out = len(pd.read_csv(output_path, nrows=0).columns)
    if max_cols and (n_in > max_cols or n_out > max_cols):
        raise SkipExample(f"too many columns (input {n_in}, output {n_out}, limit {max_cols})")

    rows = {"max_rows": wide_rows} if wide_cols and max(n_in, n_out) > wide_cols else {}
    prompt = JUDGE_TEMPLATE.format(
        input_table=table_markdown(input_path, **rows),
        output_table=table_markdown(output_path, **rows),
    )

    score, reason = parse_reply(call_llm(client, prompt))
    if score is None:
        reason = "FAILED: model reply had no 'Score: 0/1' line"
    return score, reason


def load_previous(path, skip_failed):
    rows, order, done = {}, [], set()
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        return rows, order, done

    previous = pd.read_csv(path, dtype=str).fillna("")
    for _, prev in previous.iterrows():
        eid = str(prev["example_id"])
        rec = {k: str(prev[k]) if k in previous.columns else "" for k in FIELDNAMES}
        rows[eid] = rec
        order.append(eid)
        score, reason = rec["blind_score"].strip(), rec["blind_reason"].strip()
        if (score == "1"
                or (score and reason)
                or reason.startswith("SKIPPED")
                or (reason.startswith("FAILED") and skip_failed)):
            done.add(eid)
    print(f"Resuming: {len(done)} examples already scored in {path}")
    return rows, order, done


def save_results(path, rows, order):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for eid in order:
            writer.writerow({k: rows[eid].get(k, "") for k in FIELDNAMES})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Blind LLM judge for TabTrans input/output table pairs")
    parser.add_argument("--manifest", required=True, help="CSV with example_id, track, input_csv, output_csv")
    parser.add_argument("--out", required=True, help="results CSV")
    parser.add_argument("--limit", type=int, default=None, help="only judge the first N rows")
    parser.add_argument("--max-cols", type=int, default=50,
                        help="skip tables with more columns than this (0 disables)")
    parser.add_argument("--wide-cols", type=int, default=40,
                        help="tables with more columns than this are sent with --wide-rows rows (0 disables)")
    parser.add_argument("--wide-rows", type=int, default=30)
    parser.add_argument("--skip-failed", action="store_true", help="do not retry rows marked FAILED")
    parser.add_argument("--max-consecutive-failures", type=int, default=20,
                        help="stop after this many failures in a row (0 disables)")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    data_dir = os.path.dirname(os.path.abspath(args.manifest))

    rows, order, done = load_previous(args.out, args.skip_failed)
    for _, mrow in manifest.iterrows():
        eid = str(mrow["example_id"])
        if eid not in rows:
            rows[eid] = {"example_id": eid, "track": mrow.get("track", ""),
                         "blind_score": "", "blind_reason": ""}
            order.append(eid)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    client = make_client()
    n_failed = n_skipped = consecutive_failures = 0

    try:
        for i, row in manifest.iterrows():
            eid = str(row["example_id"])
            if eid in done:
                continue
            try:
                score, reason = judge_one(client, row, data_dir,
                                          args.max_cols, args.wide_cols, args.wide_rows)
                consecutive_failures = 0
            except SkipExample as e:
                score, reason = "", f"SKIPPED: {e}"
                n_skipped += 1
            except Exception as e:
                msg = " ".join(f"{type(e).__name__}: {e}".split())[:300]
                print(f"[WARN] example {eid} failed: {msg}", file=sys.stderr)
                score, reason = "", f"FAILED: {msg}"
                n_failed += 1
                consecutive_failures += 1

            rows[eid] = {"example_id": eid, "track": row.get("track", ""),
                         "blind_score": score, "blind_reason": reason}
            print(f"[{i + 1}/{len(manifest)}] {eid} -> blind_score={score}")
            save_results(args.out, rows, order)

            if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
                print(f"[STOP] {consecutive_failures} failures in a row. "
                      "Check the API key or server, then re-run.", file=sys.stderr)
                break
    finally:
        save_results(args.out, rows, order)

    print(f"Done. Results written to {args.out}")
    print(f"Skipped (too many columns): {n_skipped}   failed this run: {n_failed}")


if __name__ == "__main__":
    main()
