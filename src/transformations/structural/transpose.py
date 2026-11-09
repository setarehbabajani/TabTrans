from .base_structural import StructuralTransformation
from .registry import register


@register
class Transpose(StructuralTransformation):
    name = "transpose"

    def apply(self, df, column_tags, params=None):
        new_df = df.T
        new_df = new_df.reset_index(drop=True)
        new_df.columns = [f"column_{i+1}" for i in range(new_df.shape[1])]

        self.last_provenance = {
            "consumed": list(df.columns),
            "created": list(new_df.columns),
            "kept": [],
        }

        return new_df, None, None
