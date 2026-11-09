from .base_structural import StructuralTransformation
from .registry import register


@register
class Unpivot(StructuralTransformation):
    name = "unpivot"

    NAME_COL = "variable"
    VALUE_COL = "value"

    def _usable(self, df, value_cols):
        cols = [c for c in (value_cols or []) if c in df.columns]
        if len(cols) < 2:
            return None
        if len(cols) >= len(df.columns):
            return None
        return cols

    def check_for_columns(self, df, column_tags, value_cols):
        return self._usable(df, value_cols) is not None

    def apply(self, df, column_tags, params=None, value_cols=None):
        if not isinstance(params, list):
            params = [{} for _ in column_tags]
        if len(params) < len(column_tags):
            params += [{}] * (len(column_tags) - len(params))

        cols = self._usable(df, value_cols)
        if cols is None:
            print("[UNPIVOT] no usable value columns")
            return df, column_tags, params

        print("[UNPIVOT] value_cols:", cols)
        context_cols = [c for c in df.columns if c not in cols]
        print("[UNPIVOT] context cols:", context_cols)

        new_df = df.melt(
            id_vars=context_cols,
            value_vars=cols,
            var_name=self.NAME_COL,
            value_name=self.VALUE_COL,
        )
        new_df = new_df[
            new_df[self.VALUE_COL].notna() & (new_df[self.VALUE_COL] != "None")
        ]

        new_tags = []
        new_params = []

        for col in context_cols:
            idx = df.columns.get_loc(col)
            new_tags.append(column_tags[idx])
            new_params.append(params[idx])

        new_tags.append(None)
        new_params.append({})

        value_tags = [column_tags[df.columns.get_loc(c)] for c in cols]
        if len(set(value_tags)) == 1:
            new_tags.append(value_tags[0])
            new_params.append(params[df.columns.get_loc(cols[0])])
        else:
            new_tags.append(None)
            new_params.append({})

        assert len(new_tags) == len(new_df.columns), (
            f"tag/column mismatch: {len(new_tags)} tags vs {len(new_df.columns)} columns"
        )

        self.last_provenance = {
            "consumed": list(cols),
            "created": [self.NAME_COL, self.VALUE_COL],
            "kept": list(context_cols),
            "protected": [self.NAME_COL, self.VALUE_COL],
        }

        return new_df, new_tags, new_params
