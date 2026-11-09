from .base_structural import StructuralTransformation
from .registry import register


@register
class DropDuplicates(StructuralTransformation):
    name = "drop_duplicates"

    def check(self, df, column_tags, params=None):
        return df.duplicated().any() or df.T.duplicated().any()

    def apply(self, df, column_tags, params=None):
        if params is None:
            params = [{} for _ in column_tags]

        df = df.drop_duplicates()

        mask = ~df.T.duplicated()
        new_df = df.loc[:, mask]

        new_tags = [
            tag for tag, keep in zip(column_tags, mask) if keep
        ]

        new_params = [
            param for param, keep in zip(params, mask) if keep
        ]

        return new_df, new_tags, new_params
