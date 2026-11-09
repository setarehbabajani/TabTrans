import json

import pandas as pd

from config import STRUCTURAL_REVEAL_ROWS, SMALL_INPUT_ROWS, SMALL_INPUT_REVEAL_ROWS

STRUCTURAL_GLOSSARY = """
You may need to apply one or more of the following table operations. Definitions:

- PIVOT: turn distinct values of one column into new columns, filling them with
  values from another column (optionally aggregated if multiple rows share a key).
- UNPIVOT / MELT: turn multiple columns into two columns, one holding the former
  column names, one holding their values.
- TRANSPOSE: swap rows and columns -- what were column headers become the first
  column's values, and vice versa.
- GROUPBY AGGREGATE: collapse rows that share the same value(s) in one or more
  columns into a single row per group, combining the remaining columns with an
  aggregate (sum, mean, count, min, max, concat, etc.).
- SORT: reorder rows by one or more columns.
- CLEANUP: drop duplicate rows, drop duplicate columns, or drop columns that
  are mostly null (few enough non-null values to be uninformative).
- DIVIDE-MIXED: when a single column contains values of different data types
  across its cells (e.g. some numeric, some text, some dates), split it into
  one column per data type, placing each cell's value in the column matching
  its type and leaving the others null for that row.

""".strip()

INTRO_WITH_GLOSSARY = (
    "You are given a table transformation task.\n\n"
    + STRUCTURAL_GLOSSARY
    + "\n\nYou are not told which of these were used, in what order, or how many -- infer\n"
    "that from the input table and the sample output rows below."
)
INTRO_PLAIN = (
    "You are given a table transformation task.\n\n"
    "You are not told which transformation was applied -- infer it from the input\n"
    "table and the sample output rows below."
)

GLOSSARY_TYPES = ("structural", "mixed")
TRACKS = ("structural", "mixed", "single_column")


def effective_version(composition_type: str, track: str = "auto") -> str:
    kind = (composition_type or "").strip().lower() if track in (None, "", "auto") else track
    return "glossary" if kind in GLOSSARY_TYPES else "plain"


def _rows_to_json(df: pd.DataFrame) -> str:
    clean = df.astype(object).where(pd.notnull(df), None)
    return json.dumps(clean.to_dict(orient="records"), ensure_ascii=False)


def reveal_count(n_input_rows: int, n_gold_rows: int) -> int:
    m = SMALL_INPUT_REVEAL_ROWS if n_input_rows <= SMALL_INPUT_ROWS else STRUCTURAL_REVEAL_ROWS
    return max(0, min(m, n_gold_rows))


def build_prompt(input_df: pd.DataFrame, gold_df: pd.DataFrame, n_reveal: int,
                 has_orig_index: bool, include_glossary: bool) -> str:
    cols = [c for c in gold_df.columns if c != "orig_index"]
    if "orig_index" in gold_df.columns:
        cols.append("orig_index")
    partial = gold_df.iloc[:n_reveal][cols]
    cols_json = json.dumps([str(c) for c in cols], ensure_ascii=False)

    orig_note = (
        "\nThe last column, orig_index, is the original index of that row in "
        "the input table above.\n"
    ) if has_orig_index else ""

    intro = INTRO_WITH_GLOSSARY if include_glossary else INTRO_PLAIN

    return f"""{intro}

Full input table ({len(input_df)} rows):
{_rows_to_json(input_df)}

A SMALL PARTIAL sample of the output table -- only {len(partial)} row(s)
out of what should be a much larger output. These are shown ONLY so you can
figure out the output schema and the transformation logic.
Output columns: {cols_json}
{_rows_to_json(partial)}
{orig_note}
Now produce the COMPLETE output table: apply the same transformation to the
whole input table above ({len(input_df)} rows), using exactly the same output
columns shown in the sample rows (do not omit or rename any of them).

Return ONLY a JSON array of output row objects. No prose, no markdown fences."""
