from .base_structural import StructuralTransformation
from .registry import register

@register
class WrapMultiHeader(StructuralTransformation):
    name = "wrap_multi_header"

    def check(self, df, column_tags, params=None):
        return False

    def apply(self, df, column_tags, params=None):
        new_df = df.copy(deep=True)
        header = [str(c).strip() for c in new_df.columns]
        first_row = [str(v).strip() for v in new_df.iloc[0]]
        new_cols = []

        for h, v in zip(header, first_row):
            h = h.strip()
            v = v.strip()

            if v.lower() in {"nan", ""}:
                new_cols.append(h)

            elif h.lower() in {"nan", ""}:
                new_cols.append(v)

            else:
                new_cols.append(f"{h}_{v}")

        new_df.columns = new_cols

        new_df = new_df.iloc[1:].reset_index(drop=True)

        return new_df, column_tags, params
