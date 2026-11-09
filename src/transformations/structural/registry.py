STRUCTURAL_TRANSFORMATIONS = []

def register(transformation):
    STRUCTURAL_TRANSFORMATIONS.append(transformation)
    return transformation

def get_all():
    return [t() for t in STRUCTURAL_TRANSFORMATIONS]
