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
class DropRows(StructuralTransformation):
    name = "drop_rows"

    def check(
        self,
        df,
        column_tags,
        params=None
    ):
        threshold = 0.5

        missing_mask = (

            df.isna()

            |

            df.applymap(

                lambda x:
                str(x)
                .strip()
                .lower()

                in MISSING_TOKENS
            )
        )

        row_missing = (
            missing_mask.mean(
                axis=1
            )
        )

        return (
            row_missing
            > threshold
        ).any()

    def apply(
        self,
        df,
        column_tags,
        params=None
    ):
        threshold = 0.5

        if params is None:
            params = [
                {}
                for _ in column_tags
            ]

        missing_mask = (

            df.isna()

            |

            df.applymap(

                lambda x:
                str(x)
                .strip()
                .lower()

                in MISSING_TOKENS
            )
        )

        keep_mask = (

            missing_mask.mean(
                axis=1
            )
            <= threshold
        )

        new_df = (
            df[keep_mask]
            .reset_index(
                drop=True
            )
        )

        return (
            new_df,
            column_tags,
            params
        )
