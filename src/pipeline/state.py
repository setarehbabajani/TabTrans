
class TableState:
    def __init__(self, df, tags, steps=None,
                 structural_count=0,
                 column_counts=None,
                 structural_used=None,
                 column_history=None,
                 split_groups=None,
                 merged_columns=None):
        self.df = df
        self.tags = tags
        self.steps = steps or []

        self.structural_count = structural_count
        self.column_counts = column_counts or {}
        self.structural_used = structural_used or set()
        self.column_history = column_history
        self.split_groups = split_groups if split_groups is not None else {}
        self.merged_columns = merged_columns if merged_columns is not None else set()

    def signature(self):
        return (
            tuple(self.df.fillna("").infer_objects(copy=False).astype(str).values.flatten()),
            tuple(col["tag"] for col in self.tags["columns"])
        )
