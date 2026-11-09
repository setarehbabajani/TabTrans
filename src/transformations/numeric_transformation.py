import numpy as np
import pandas as pd
from .base import Transformation


class NumericTransformation(Transformation):
    name = "numeric_transformation"
    input_arity = "n"
    output_arity = "n"

    def check(self, columns, tags, params):
        if tags == ["YEAR", "YEAR"]:
            return True
        if tags == ["YEAR"]:
            return True
        if tags[0] == "NUMERIC" and len(tags) == 1:
            col_params = (
                params[0]
                if isinstance(params, list) and params
                else (params if isinstance(params, dict) else {})
            )
            thousand_sep = col_params.get("thousand_separator")
            decimal_sep = col_params.get("decimal_separator")
            already_normalized = (
                thousand_sep is None
                and (decimal_sep is None or decimal_sep == ".")
            )
            return already_normalized
        return False

    def apply(self, columns, tags, params):
        if tags == ["YEAR", "YEAR"]:
            def extract_aligned(a, b):
                a_digits = ''.join(filter(str.isdigit, str(a)))
                b_digits = ''.join(filter(str.isdigit, str(b)))

                if not a_digits or not b_digits:
                    return None, None

                a_val = int(a_digits)

                if len(b_digits) == len(a_digits):
                    return a_val, int(b_digits)
                prefix = a_digits[:-len(b_digits)]
                new_b = int(prefix + b_digits)
                if new_b < a_val:
                    new_b += 10 ** len(b_digits)
                return a_val, new_b

            aligned_start = []
            aligned_end = []
            for a, b in zip(columns[0], columns[1]):
                a_val, b_val = extract_aligned(a, b)

                if a_val is None or b_val is None:
                    aligned_start.append(float("nan"))
                    aligned_end.append(float("nan"))
                else:
                    aligned_start.append(a_val)
                    aligned_end.append(b_val)

            start = pd.Series(aligned_start)
            end = pd.Series(aligned_end)

            new_col = end - start
            new_params = {"thousand_separator": None, "decimal_separator": None}
            self.last_used_params = {"mode": "year_difference"}
            return [new_col], ["NUMERIC"], new_params

        if tags == ["YEAR"]:
            col = columns[0]

            def century_prefix(value):
                digits = ''.join(filter(str.isdigit, str(value)))
                if len(digits) != 4:
                    return digits
                return digits[:2]

            new_col = col.apply(century_prefix)

            has_leading_zero = bool(
                new_col.dropna().astype(str).str.startswith("0").any()
            )

            self.last_used_params = {
                "mode": "year_century_prefix",
                "has_leading_zero": has_leading_zero,
            }
            return (
                [new_col],
                ["IDENTIFIER_CODE"],
                [{
                    "code_type": "year_century_prefix",
                    "has_leading_zero": has_leading_zero,
                }],
            )

        col = pd.to_numeric(columns[0], errors="coerce")

        cumulative = col.cumsum()

        exponential = np.exp(col)

        square_root = np.sqrt(col)

        mean_centered = col - col.mean()

        self.output_name_hints = ["cumsum", "exp", "sqrt", "demeaned"]
        self.last_used_params = {
            "mode": "numeric_transforms",
            "ops": ["cumsum", "exp", "sqrt", "demeaned"],
        }

        return (
            [cumulative, exponential, square_root, mean_centered],
            [
                "IDENTIFIER_CODE",
                "IDENTIFIER_CODE",
                "IDENTIFIER_CODE",
                "IDENTIFIER_CODE",
            ],
            [
                {
                    "code_type": "numeric_cumulative_sum",
                    "has_leading_zero": False,
                },
                {
                    "code_type": "numeric_exponential",
                    "has_leading_zero": False,
                },
                {
                    "code_type": "numeric_square_root",
                    "has_leading_zero": False,
                },
                {
                    "code_type": "numeric_mean_centered",
                    "has_leading_zero": False,
                }
            ],
        )
