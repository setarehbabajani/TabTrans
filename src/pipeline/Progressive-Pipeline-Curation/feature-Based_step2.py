from __future__ import annotations

import numpy as np

CONTINUOUS_FEATURES = [
    "semantic_score",
    "phase1_structural_score",
    "phase3_structural_score",
    "root_coverage_ratio",
    "max_root_depth",
    "total_depth",
    "cleanup_score",
    "diversity",
]

CATEGORICAL_FEATURES = [
    "dependency_type",
    "phase1_structural_identity",
]


LOW_QUANTILE = 0.33
HIGH_QUANTILE = 0.66


def assign_tier(value: float, low_q: float, high_q: float) -> str:
    if value <= low_q:
        return "low"
    if value <= high_q:
        return "medium"
    return "high"


def build_regions(outputs: list[dict]) -> list[dict]:
    feature_thresholds: dict[str, tuple[float, float]] = {}

    for feature in CONTINUOUS_FEATURES:
        values = [o["features"][feature] for o in outputs]
        feature_thresholds[feature] = (
            float(np.quantile(values, LOW_QUANTILE)),
            float(np.quantile(values, HIGH_QUANTILE)),
        )

    for o in outputs:
        regions: set[tuple] = set()

        for feature in CONTINUOUS_FEATURES:
            value = o["features"][feature]
            low_q, high_q = feature_thresholds[feature]
            regions.add((feature, assign_tier(value, low_q, high_q)))

        for feature in CATEGORICAL_FEATURES:
            regions.add((feature, o["features"][feature]))

        o["regions"] = regions

    return outputs
