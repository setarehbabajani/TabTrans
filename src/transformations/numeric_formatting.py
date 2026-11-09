import pandas as pd
from .base import Transformation


MISSING_TOKENS = {

    "",
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


class NumericFormatting(Transformation):
    name = "numeric_formatting"
    input_arity = 1
    output_arity = 1

    def check(
        self,
        columns,
        tags,
        params
    ):
        if tags[0] != "NUMERIC":
            return False

        col = columns[0]

        null_ratio = (

            col.isna()

            |

            col.astype(str)
            .str.strip()
            .str.lower()
            .isin(
                MISSING_TOKENS
            )

        ).mean()

        if null_ratio > 0.6:
            return False

        thousand_sep = params.get(
            "thousand_separator"
        )

        decimal_sep = params.get(
            "decimal_separator"
        )

        return (

            (thousand_sep is not None)

            or

            (
                decimal_sep is not None
                and decimal_sep != "."
            )
        )

    def apply(
        self,
        columns,
        tags,
        params
    ):
        col = columns[0].copy()

        tag = tags[0]

        thousand_sep = params.get(
            "thousand_separator"
        )

        decimal_sep = params.get(
            "decimal_separator",
            "."
        )

        if (
            thousand_sep is not None
            or
            decimal_sep is not None
        ):
            def convert(text):
                if pd.isna(text):
                    return None

                text = (
                    str(text)
                    .strip()
                )

                if (
                    text.lower()
                    in MISSING_TOKENS
                ):
                    return None

                if thousand_sep:
                    text = text.replace(
                        thousand_sep,
                        ""
                    )

                if (
                    decimal_sep
                    and
                    decimal_sep != "."
                ):
                    text = text.replace(
                        decimal_sep,
                        "."
                    )

                try:
                    if "." in text:
                        return float(text)

                    return int(text)

                except:
                    return None

            col = col.apply(
                convert
            )

            new_params = {

                "thousand_separator":
                    None,

                "decimal_separator":
                    "."
            }

        else:
            new_params = (
                params.copy()
            )

        if tag == "NUMERIC":
            col = col.apply(

                lambda x:
                round(
                    float(x),
                    2
                )

                if pd.notnull(x)

                else x
            )

        return (
            [col],
            ["NUMERIC"],
            [new_params]
        )
