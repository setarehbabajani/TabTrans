import pandas as pd

from .base_structural import StructuralTransformation
from .registry import register


@register
class SortRows(StructuralTransformation):
    name = "sort_rows"

    def check(self, df, column_tags, params=None):
        priority = [
            "DATE_ISO_YYYY_MM_DD",
            "NUMERIC",
            "STRING"
        ]

        for tag in priority:
            for idx, col_tag in enumerate(column_tags):
                if col_tag == tag:
                    col_name = df.columns[idx]

                    try:
                        if tag == "NUMERIC":
                            series = pd.to_numeric(df[col_name], errors="coerce")
                        elif tag == "DATE_ISO_YYYY_MM_DD":
                            series = pd.to_datetime(df[col_name], errors="coerce")
                        else:
                            series = df[col_name].astype(str)

                        series = series.dropna()

                        if not series.is_monotonic_increasing:
                            return True

                    except Exception:
                        continue

        return False

    @staticmethod
    def _unique_orig_index_name(columns):
        name = "orig_index"
        if name not in columns:
            return name
        i = 2
        while f"{name}_{i}" in columns:
            i += 1
        return f"{name}_{i}"

    def apply(self, df, column_tags, params=None):
        if params is None:
            params = [{} for _ in column_tags]

        priority = [
            "DATE_ISO_YYYY_MM_DD",
            "NUMERIC",
            "STRING"
        ]

        new_df = df.copy()

        for tag in priority:
            for idx, col_tag in enumerate(column_tags):
                if col_tag == tag:
                    col_name = new_df.columns[idx]

                    try:
                        if tag == "NUMERIC":
                            series = pd.to_numeric(new_df[col_name], errors="coerce")
                        elif tag == "DATE_ISO_YYYY_MM_DD":
                            series = pd.to_datetime(new_df[col_name], errors="coerce")
                        else:
                            series = new_df[col_name].astype(str)

                        series = series.dropna()

                        if not series.is_monotonic_increasing:
                            orig_col = self._unique_orig_index_name(new_df.columns)
                            new_df[orig_col] = range(len(new_df))

                            new_df = new_df.sort_values(
                                by=col_name
                            ).reset_index(drop=True)

                            new_tags = list(column_tags) + ["IDENTIFIER_CODE"]
                            new_params = list(params) + [
                                {
                                    "code_type": "orig_index",
                                    "has_leading_zero": False,
                                }
                            ]

                            return new_df, new_tags, new_params

                    except Exception:
                        continue

        return new_df, column_tags, params
