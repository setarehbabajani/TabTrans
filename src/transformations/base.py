class Transformation:
    name = ""
    input_arity = None
    output_arity = None

    def check(self, columns, tags, params):
        return True

    def apply(self, columns, tags, params):
        raise NotImplementedError
