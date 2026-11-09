from __future__ import annotations

from .llm_structural_check import N_HEAD_ROWS

CACHE_NAME = "groupby"

SYSTEM_PROMPT = (
    "You are a meticulous data-wrangling assistant. You analyze small "
    "samples of tabular data and judge whether a column is a meaningful "
    "grouping key for aggregation. You always answer with strict, valid "
    "JSON only — no markdown fences, no commentary outside the JSON object."
)

DEFINITION = """Definition of the transformation:

GROUPBY + AGGREGATE: One column's repeated values define row groups; a
numeric column is then aggregated (mean/sum/count) within each group,
collapsing the table to one row per group.

Mark groupby applicable for a column only when its values form meaningful,
repeated categories that a person would actually want to summarise by. Mark it NOT applicable
when the values are mostly unique row identifiers, because grouping by them collapses nothing and produces a
table that is the same size as the input.

Be conservative: answer true only when the grouping is clearly meaningful.
Prefer false when uncertain.
"""


def build_groupby_prompt(table_name: str, table_preview: str) -> str:
    return f"""{DEFINITION}
Below is table "{table_name}": the column headers and {N_HEAD_ROWS} sample
data rows.

{table_preview}

TASK:
For EACH column header above, decide:
  - "groupby": would grouping the table by this column and aggregating a
    numeric column produce a meaningful summary?
Give a one-sentence reason for each decision.

Respond with ONLY a single JSON object in exactly this shape (no markdown
fences, no extra text):

{{
  "columns": {{
    "<column_name>": {{
      "groupby": {{"applicable": false, "reason": "..."}}
    }}
  }}
}}

Use the exact header strings as keys under "columns". Include every column,
in the same order as the header row. Use boolean true/false only (not
strings). Do not add any extra top-level keys.
"""


def parse_groupby_response(response: dict) -> dict:
    column_flags: dict[str, dict] = {}
    columns = (response or {}).get("columns")
    if isinstance(columns, dict):
        for col_name, info in columns.items():
            if not isinstance(info, dict):
                continue
            block = info.get("groupby")
            applicable = None
            if isinstance(block, dict) and "applicable" in block:
                applicable = bool(block["applicable"])
            column_flags[str(col_name)] = {
                "pivot": None,
                "unpivot": None,
                "groupby": applicable,
            }

    return {
        "transpose_needed": None,
        "column_flags": column_flags,
        "multi_header_wrapped": False,
    }
