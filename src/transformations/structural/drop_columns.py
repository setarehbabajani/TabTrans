from .base_structural import StructuralTransformation
from .registry import register

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

@register
class DropColumns(StructuralTransformation):
    name = "drop_columns"

    def check(
        self,
        df,
        column_tags,
        params=None
    ):
        threshold = 0.6

        for col in df.columns:
            missing_mask = (

                df[col].isna()

                |

                df[col].apply(

                    lambda x:
                    str(x)
                    .strip()
                    .lower()

                    in MISSING_TOKENS
                )
            )

            if (
                missing_mask.mean()
                > threshold
            ):
                return True

        return False

    def apply(
        self,
        df,
        column_tags,
        params=None
    ):
        threshold = 0.6

        if params is None:
            params = [
                {}
                for _ in column_tags
            ]

        keep_cols = []
        keep_tags = []
        keep_params = []

        for col, tag, par in zip(
            df.columns,
            column_tags,
            params
        ):
            missing_mask = (

                df[col].isna()

                |

                df[col].apply(

                    lambda x:
                    str(x)
                    .strip()
                    .lower()

                    in MISSING_TOKENS
                )
            )

            if (
                missing_mask.mean()
                <= threshold
            ):
                keep_cols.append(col)
                keep_tags.append(tag)
                keep_params.append(par)

        new_df = df[
            keep_cols
        ].copy()

        return (
            new_df,
            keep_tags,
            keep_params
        )
