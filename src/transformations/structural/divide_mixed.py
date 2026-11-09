import pandas as pd
import re

from .base_structural import StructuralTransformation
from .registry import register

NUMERIC_REGEX = r"^\d{1,3}(,\d{3})*(\.\d+)?$|^\d+(\.\d+)?$"

UNPIVOT_VALUE_COL = "value"


def _parse_numeric(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None


@register
class DivideMixed(StructuralTransformation):
    name = "divide_mixed"

    def check(self, df, column_tags, params=None):
        if "COLUMN_MIXED" not in column_tags:
            return False

        if params is None:
            return False

        for idx, tag in enumerate(column_tags):
            if tag != "COLUMN_MIXED":
                continue

            if str(df.columns[idx]) == UNPIVOT_VALUE_COL:
                continue

            if isinstance(params, list):
                if idx >= len(params) or params[idx] is None:
                    continue
                col_params = params[idx]
            else:
                col_params = params

            has_numeric = col_params.get("has_numeric", False)
            has_string = col_params.get("has_string", False)
            has_boolean = col_params.get("has_boolean", False)
            type_count = sum([has_numeric, has_string, has_boolean])

            if type_count >= 2:
                return True

        return False

    def apply(self, df, column_tags, params):
        if "COLUMN_MIXED" not in column_tags:
            return df, column_tags, params

        new_columns_data = {}
        new_tags = []
        new_params = []
        processed_any = False

        for idx in range(len(df.columns)):
            col_name = df.columns[idx]
            col = df.iloc[:, idx]
            tag = column_tags[idx]
            col_params = params[idx] if idx < len(params) and params[idx] else {}

            if tag != "COLUMN_MIXED" or str(col_name) == UNPIVOT_VALUE_COL:
                new_columns_data[col_name] = col
                new_tags.append(tag)
                new_params.append(col_params)
                continue

            has_numeric = col_params.get("has_numeric", False)
            has_string = col_params.get("has_string", False)
            has_boolean = col_params.get("has_boolean", False)
            type_count = sum([has_numeric, has_string, has_boolean])

            if type_count < 2:
                new_columns_data[col_name] = col
                new_tags.append(tag)
                new_params.append(col_params)
                continue

            processed_any = True

            print(f"Column index: {idx}")
            print(f"Column name: {col_name}")
            print("Sample values:", col.head(10).tolist())

            numeric_col = None
            string_col = None
            boolean_col = None

            if has_numeric:
                numeric_col = col.apply(
                    lambda x: _parse_numeric(x)
                    if isinstance(x, (int, float)) or re.match(NUMERIC_REGEX, str(x))
                    else None
                )
                print("Numeric non-null count:", numeric_col.notna().sum())

            if has_boolean:
                boolean_col = col.apply(
                    lambda x: True
                    if str(x).lower() in {"true", "yes", "y", "1"}
                    else False
                    if str(x).lower() in {"false", "no", "n", "0"}
                    else None
                )
                print("Boolean non-null count:", boolean_col.notna().sum())

            if has_string:
                string_col = col.apply(
                    lambda x: x
                    if (
                        x is None
                        or str(x).strip() == ""
                        or (
                            not re.match(NUMERIC_REGEX, str(x))
                            and str(x).lower() not in {"true", "false", "yes", "no", "1", "0"}
                        )
                    )
                    else None
                )
                print("String non-null count:", string_col.notna().sum())

            if has_numeric:
                has_comma = col.astype(str).str.contains(",").any()
                new_columns_data[f"{col_name}_numeric"] = numeric_col
                new_tags.append("NUMERIC")
                new_params.append({
                    "thousand_separator": "," if has_comma else None,
                    "decimal_separator": ".",
                })

            if has_boolean:
                new_columns_data[f"{col_name}_boolean"] = boolean_col
                new_tags.append("BOOLEAN_VARIATION")
                new_params.append({})

            if has_string:
                new_columns_data[f"{col_name}_string"] = string_col
                new_tags.append("STRING")
                new_params.append({})

        if not processed_any:
            return df, column_tags, params

        new_df = pd.DataFrame(new_columns_data, index=df.index)

        print("Final columns:", new_df.columns.tolist())

        return new_df, new_tags, new_params
