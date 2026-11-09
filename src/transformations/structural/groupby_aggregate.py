from .base_structural import StructuralTransformation
from .registry import register
import pandas as pd


def _is_normalized_numeric(tag, col_params):
    if tag != "NUMERIC":
        return False
    if not isinstance(col_params, dict):
        return True
    thousand_sep = col_params.get("thousand_separator")
    decimal_sep = col_params.get("decimal_separator")
    return thousand_sep is None and (decimal_sep is None or decimal_sep == ".")


@register
class GroupByAggregate(StructuralTransformation):
    name = "groupby_aggregate"

    def check(self, df, column_tags, params=None):
        if not isinstance(params, list):
            params = [{} for _ in column_tags]

        numeric_cols = [
            col
            for col, tag, p in zip(df.columns, column_tags, params)
            if _is_normalized_numeric(tag, p)
        ]
        if not numeric_cols:
            return False

        candidate_cols = [
            col for col, tag in zip(df.columns, column_tags) if tag != "NUMERIC"
        ]
        if not candidate_cols:
            return False

        group_col = self.select_group_column(df, column_tags)
        if group_col is None:
            return False

        counts = df[group_col].value_counts(dropna=True)
        if (counts > 1).sum() == 0:
            return False

        return True

    def select_group_column(self, df, column_tags):
        candidate_cols = [
            col
            for col, tag in zip(df.columns, column_tags)
            if tag not in ("NUMERIC", "IDENTIFIER_CODE")
        ]
        if not candidate_cols:
            return None

        def grouping_score(col):
            counts = df[col].value_counts(dropna=True)
            if len(counts) == 0:
                return -float("inf")
            non_singleton = (counts > 1).sum()
            singleton = (counts == 1).sum()
            return non_singleton - 0.5 * singleton

        return max(candidate_cols, key=grouping_score)

    def select_numeric_column(self, df, column_tags, group_col=None, params=None):
        if params is None:
            for col, tag in zip(df.columns, column_tags):
                if tag == "NUMERIC" and col != group_col:
                    return col
            return None

        if len(params) < len(column_tags):
            params = list(params) + [{}] * (len(column_tags) - len(params))

        for col, tag, p in zip(df.columns, column_tags, params):
            if col == group_col:
                continue
            if _is_normalized_numeric(tag, p):
                return col
        return None

    def apply(self, df, column_tags, params=None, group_col=None):
        if not isinstance(params, list):
            params = [{} for _ in column_tags]

        if len(params) < len(column_tags):
            params += [{}] * (len(column_tags) - len(params))

        if group_col is None:
            group_col = self.select_group_column(df, column_tags)
        if group_col is None or group_col not in df.columns:
            return df, column_tags, params

        numeric_col = self.select_numeric_column(df, column_tags, group_col, params)
        if numeric_col is None:
            return df, column_tags, params

        group_tag = column_tags[df.columns.get_loc(group_col)]
        group_param = params[df.columns.get_loc(group_col)]

        print(f"[GROUPBY] group_col={group_col!r}  aggregate={numeric_col!r} (mean)")

        numeric_series = pd.to_numeric(df[numeric_col], errors="coerce")
        tmp_df = df.copy()
        tmp_df[numeric_col] = numeric_series

        new_df = (
            tmp_df.groupby(group_col)[numeric_col]
            .mean()
            .reset_index(name=f"{numeric_col}_mean")
        )

        new_tags = [group_tag, "NUMERIC"]
        new_params = [
            group_param,
            {"thousand_separator": None, "decimal_separator": "."},
        ]

        self.last_provenance = {
            "consumed": [group_col, numeric_col],
            "created": [f"{numeric_col}_mean"],
            "kept": [group_col],
        }

        return new_df, new_tags, new_params
