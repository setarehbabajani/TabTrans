import re

import pandas as pd

from .base_structural import StructuralTransformation
from .registry import register


_LEADING_NUMBER_RE = re.compile(r"[+-]?[0-9][0-9,]*\.?[0-9]*")


def _to_numeric_series(series):
    def parse_one(value):
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        match = _LEADING_NUMBER_RE.match(text)
        if not match or match.group(0) in ("", "-", "+"):
            return None

        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None

    return series.apply(parse_one)


@register
class Pivot(StructuralTransformation):
    name = "pivot"

    def check_for_column(self, df, column_tags, column_col, exclude_cols=None):
        return self.select_pivot_roles(
            df, column_tags, column_col=column_col, exclude_cols=exclude_cols
        ) is not None

    def select_pivot_roles(self, df, column_tags, column_col=None, exclude_cols=None):
        if column_col is None:
            return None
        if column_col not in df.columns:
            return None

        blocked = {str(c) for c in (exclude_cols or [])}
        blocked.add(str(column_col))

        numeric_cols = [
            col
            for col, tag in zip(df.columns, column_tags)
            if tag == "NUMERIC" and df[col].notna().sum() > 0
        ]
        value_candidates = [c for c in numeric_cols if str(c) not in blocked]
        value_tag = "NUMERIC"

        if not value_candidates:
            unit_cols = [
                col
                for col, tag in zip(df.columns, column_tags)
                if tag == "NUM_WITH_UNIT" and df[col].notna().sum() > 0
            ]
            value_candidates = [c for c in unit_cols if str(c) not in blocked]
            value_tag = "NUM_WITH_UNIT"

        if not value_candidates:
            return None
        value_col = value_candidates[0]

        index_cols = [
            col for col in df.columns if col not in (column_col, value_col)
        ]
        if not index_cols:
            return None

        return {
            "column_col": column_col,
            "value_col": value_col,
            "value_tag": value_tag,
            "index_cols": index_cols,
        }

    def apply(self, df, column_tags, params=None, column_col=None, exclude_cols=None):
        if params is None:
            params = [{} for _ in column_tags]

        roles = self.select_pivot_roles(
            df, column_tags, column_col=column_col, exclude_cols=exclude_cols
        )
        if roles is None:
            return df, column_tags, params

        column_col = roles["column_col"]
        value_col = roles["value_col"]
        value_tag = roles["value_tag"]
        index_cols = roles["index_cols"]
        print(
            f"[PIVOT] column_col (values->headers): {column_col!r}; "
            f"value_col={value_col!r} ({value_tag}); index_cols={index_cols}"
        )

        working_df = df.copy()
        if value_tag == "NUM_WITH_UNIT":
            working_df[value_col] = _to_numeric_series(working_df[value_col])

        SENTINEL = "__PIVOT_MISSING__"
        for col in index_cols:
            if working_df[col].isna().any():
                working_df[col] = working_df[col].fillna(SENTINEL)

        try:
            new_df = working_df.pivot_table(
                index=index_cols,
                columns=column_col,
                values=value_col,
                aggfunc="sum",
            ).reset_index()
        except Exception as exc:
            print(f"[PIVOT] pivot_table failed: {exc}")
            return df, column_tags, params

        for col in index_cols:
            if (new_df[col] == SENTINEL).any():
                new_df[col] = new_df[col].replace(SENTINEL, None)

        new_df.columns.name = None
        new_df.columns = [str(col) for col in new_df.columns]

        if new_df.empty or len(new_df.columns) <= len(index_cols):
            print("[PIVOT] produced no value columns")
            return df, column_tags, params

        new_tags = []
        new_params = []
        for col in new_df.columns:
            if col in index_cols:
                idx = df.columns.get_loc(col)
            else:
                idx = df.columns.get_loc(value_col)
            new_tags.append(column_tags[idx])
            new_params.append(params[idx])

        self.last_provenance = {
            "consumed": [column_col, value_col],
            "created": [c for c in new_df.columns if c not in index_cols],
            "kept": list(index_cols),
        }

        return new_df, new_tags, new_params
