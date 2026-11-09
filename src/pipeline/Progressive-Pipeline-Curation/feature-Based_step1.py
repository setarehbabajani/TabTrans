import re
from collections import defaultdict
import pandas as pd

cleanup_names = {
    "drop_columns",
    "drop_duplicates",
    "drop_rows"
}

MULTI_COLUMN_TRANSFORMS = {

    "concat_columns",
    "numeric_transformation",

    "transpose",
    "pivot",
    "unpivot",
    "divide_mixed",
    "groupby_aggregate"
}

TRANSFORMATION_WEIGHTS = {
    "string_normalization": 1,
    "boolean_normalization": 1,
    "numeric_transformation": 2,
    "numeric_formatting": 3,
    "date_format": 3,
    "concat_columns": 3,
    "substring": 4,
    "pattern_count": 5,
    "person_derivations": 5,
    "unit_conversion": 4,
    "split_column": 5,
}


def get_root_columns(col):
    if "__MERGED__" in col:
        roots = []

        for part in col.split("__MERGED__"):
            roots.extend(
                get_root_columns(part)
            )

        return roots

    if re.match(r"^.+_\d+$", col):
        return [
            col.rsplit("_", 1)[0]
        ]

    return [col]


def compute_dependency_type(
    phase1,
    phase2,
    phase3
):

    for step in (
        phase1
        + phase2
        + phase3
    ):
        t = step.get(
            "transformation"
        )

        if (
            t in
            MULTI_COLUMN_TRANSFORMS
        ):
            return "multi_column"

    return "single_column"


IGNORED_PHASE1_IDENTITY_TRANSFORMS = {"wrap_multi_header"}

def phase1_structural_identity_key(phase1: list) -> str:
    names = [
        step.get("transformation")
        for step in phase1
        if step.get("transformation")
        and step.get("transformation") not in IGNORED_PHASE1_IDENTITY_TRANSFORMS
    ]
    return "|".join(names) if names else "none"


def read_source_column_count(source_csv_path: str) -> int:
    return len(pd.read_csv(source_csv_path, nrows=0).columns)


def compute_features(pipeline, total_source_columns: int):
    phase1 = pipeline.get(
        "phase1_structural",
        []
    )

    phase2 = pipeline.get(
        "phase2_cell_level",
        []
    )

    phase3 = pipeline.get(
        "phase3_structural",
        []
    )

    total_depth = (
        len(phase1)
        + len(phase2)
        + len(phase3)
    )

    phase1_structural_score = len(phase1)

    phase3_structural_score = 0
    cleanup_score = 0

    for step in phase3:
        t = step.get("transformation")
        if t in cleanup_names:
            cleanup_score += 1
        else:
            phase3_structural_score += 1

    semantic_score = 0

    roots = set()

    root_depths = defaultdict(int)

    diversity = set()

    for step in phase2:
        trans = step.get(
            "transformation"
        )

        semantic_score += (
            TRANSFORMATION_WEIGHTS
            .get(trans, 1)
        )

        diversity.add(trans)

        cols = step.get(
            "columns",
            []
        )

        step_roots = set()

        for col in cols:
            extracted_roots = (
                get_root_columns(col)
            )

            for root in extracted_roots:
                roots.add(root)

                step_roots.add(root)

        for r in step_roots:
            root_depths[r] += 1

    max_root_depth = (
        max(root_depths.values())
        if root_depths else 0
    )

    dependency_type = (
        compute_dependency_type(
            phase1,
            phase2,
            phase3
        )
    )

    return {

        "total_depth":
            total_depth,

        "phase1_structural_score":
            phase1_structural_score,

        "phase3_structural_score":
            phase3_structural_score,

        "semantic_score":
            semantic_score,

        "cleanup_score":
            cleanup_score,

        "root_coverage_ratio":
            (len(roots) / total_source_columns)
            if total_source_columns > 0
            else 0,

        "max_root_depth":
            max_root_depth,

        "diversity":
            len(diversity),

        "dependency_type":
            dependency_type,

        "phase1_structural_identity":
            phase1_structural_identity_key(phase1),
    }
