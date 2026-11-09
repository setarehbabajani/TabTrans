class StructuralTransformation:
    name = None

    min_columns = 1

    def check(self, df, column_tags, params=None):
        raise NotImplementedError

    def apply(self, df, column_tags, params=None):
        raise NotImplementedError
