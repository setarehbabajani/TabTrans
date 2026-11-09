import pandas as pd
from .base import Transformation


class BooleanNormalization(Transformation):
    name = "boolean_normalization"
    input_arity = 1
    output_arity = 1

    TRUE_VALUES = {"yes", "y", "1", "true"}
    FALSE_VALUES = {"no", "n", "0", "false"}

    def check(self, columns, tags, params):
        if tags[0] != "BOOLEAN":
            return False

        if params.get("is_canonical_form") is not False:
            return False

        col = columns[0]
        null_ratio = (
            col.isna() |
            col.astype(str).str.strip().isin([
                "",
                "None",
                "none",
                "null",
                "NULL",
                "nan",
                "NaN"
            ])
        ).mean()

        if null_ratio > 0.6:
            return False
        return True

    def apply(self, columns, tags, params):
        col = columns[0]

        def normalize(x):
            if pd.isna(x):
                return None
            v = str(x).lower().strip()
            if v in self.TRUE_VALUES:
                return True
            if v in self.FALSE_VALUES:
                return False

            return None

        col = col.apply(normalize)

        out_params = dict(params)
        out_params["is_canonical_form"] = True

        return [col], ["BOOLEAN"], out_params
