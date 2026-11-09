import pandas as pd
import re
from .base import Transformation


MISSING_TOKENS = {

    "nan",
    "none",
    "null",

    "n/a",
    "na",
    "n.a",
    "n.a.",

    "-",
    "--",
    "---",

    "_",
    "__",
    "___",

    "?",
    "??",
    "???"
}


class StringNormalization(Transformation):
    name = "string_normalization"
    input_arity = 1
    output_arity = 1

    def check(
        self,
        columns,
        tags,
        params
    ):
        if tags[0] != "STRING":
            return False

        none_ratio = (

            columns[0]

            .apply(

                lambda x:
                pd.isna(x)

                or

                str(x)
                .strip()
                .lower()

                in MISSING_TOKENS
            )

            .mean()
        )

        return (
            none_ratio <= 0.6
        )

    def apply(
        self,
        columns,
        tags,
        params
    ):
        col = columns[0].copy()

        def normalize(text):
            if pd.isna(text):
                return ""

            text = str(text).strip()

            if (
                text.lower()
                in MISSING_TOKENS
            ):
                return ""

            text = text.lower()

            text = re.sub(
                r"[^\w\s]",
                " ",
                text
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            return text

        col = col.apply(
            normalize
        )

        return (
            [col],
            ["STRING"],
            params
        )
