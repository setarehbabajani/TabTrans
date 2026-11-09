from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

LOCAL_SERVED_MODEL = "qwen2.5-72b-instruct"
MODEL_SHORT_NAME = "qwen-2.5-72b"

N_HEAD_ROWS = 15
REQUEST_TIMEOUT = int(os.environ.get("STRUCTURAL_LLM_TIMEOUT", "900"))
MAX_TOKENS = int(os.environ.get("STRUCTURAL_LLM_MAX_TOKENS", "4096"))


def chat_url() -> str:
    raw = os.environ.get("STRUCTURAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1").strip().rstrip("/")
    if raw.endswith("/chat/completions"):
        return raw
    if raw.endswith("/v1"):
        return f"{raw}/chat/completions"
    return f"{raw}/v1/chat/completions"


SYSTEM_PROMPT = (
    "You are a meticulous data-wrangling assistant. You analyze small "
    "samples of tabular data and judge whether certain structural reshape "
    "operations could be meaningfully applied. You always answer with "
    "strict, valid JSON only — no markdown fences, no commentary outside "
    "the JSON object."
)

DEFINITIONS = """Definitions of the structural transformations:

1. PIVOT (long -> wide): One column's categorical values become new column
   headers; another column supplies cell values.
   Mark pivot applicable for a column only when that column has a small,
   repeating set of category labels and stacking those labels as columns would make the table wider and more readable.

2. UNPIVOT / MELT (wide -> long): Several columns represent repeated
   instances of the same variable type and should be stacked into one name column + one value column.
   Mark unpivot applicable for a column only when it is one of a group of
   parallel/sibling columns of the same kind — not when it is a unique
   identifier or unrelated field.

3. TRANSPOSE (whole table): Flip rows and columns so former column headers
   become row labels and vice versa. Applies to the ENTIRE TABLE, not one
   column. Mark transpose needed only when observations appear to run across
   columns and variables/fields run down rows (sideways layout), rather
   than the usual one-row-per-record layout.

Be conservative: answer true only when the operation is clearly justified
by the preview. Prefer false when uncertain.
"""


def _cell_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_table_preview(preview_df: pd.DataFrame) -> tuple[str, list[str]]:
    columns = [str(c) for c in preview_df.columns]
    header_line = " | ".join(columns)
    row_lines: list[str] = []
    for i in range(min(N_HEAD_ROWS, len(preview_df))):
        row = preview_df.iloc[i]
        cells = [_cell_str(row[col]) for col in preview_df.columns]
        row_lines.append(f"Row {i + 1}: " + " | ".join(cells))

    preview = "Headers:\n" + header_line + "\n\n" + "\n".join(row_lines)
    return preview, columns


def build_user_prompt(table_name: str, table_preview: str) -> str:
    return f"""{DEFINITIONS}
Below is table "{table_name}": the column headers (after any multi-header
merge) and {N_HEAD_ROWS} sample data rows.

{table_preview}

TASK:
For EACH column header above, decide:
  - "pivot": can PIVOT be meaningfully applied using this column?
  - "unpivot": can UNPIVOT be meaningfully applied using this column?
Give a one-sentence reason for each decision.

Then, for the WHOLE TABLE (not a single column), decide:
  - "transpose": is a TRANSPOSE of the whole table needed/appropriate?

Respond with ONLY a single JSON object in exactly this shape (no markdown
fences, no extra text):

{{
  "columns": {{
    "<column_name>": {{
      "pivot":   {{"applicable": true, "reason": "..."}},
      "unpivot": {{"applicable": false, "reason": "..."}}
    }}
  }},
  "table": {{
    "transpose": {{"needed": false, "reason": "..."}}
  }}
}}

Use the exact header strings as keys under "columns". Include every column,
in the same order as the header row. Use boolean true/false only (not strings).
Do not add any extra top-level keys.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text[:300]!r}")
    return json.loads(text[start : end + 1])


def call_model(system_prompt: str, user_prompt: str) -> dict:
    body = {
        "model": LOCAL_SERVED_MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    url = chat_url()
    resp = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 400:
        body.pop("response_format")
        resp = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM HTTP {resp.status_code} at {url}: {resp.text[:500]}")
    return _extract_json(resp.json()["choices"][0]["message"]["content"])


def model_output_path(wiki_path: Path) -> Path:
    return wiki_path / f"structural_transform_check.{MODEL_SHORT_NAME}.json"
