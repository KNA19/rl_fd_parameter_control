from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from envs.layout_context import LayoutContext
from features.normalizers import FloatDict, clip01, dictionary_to_vector, normalize_linear, safe_float


class LayoutFeatureExtractor:
    """
    Extract current layout-quality features A_t.

    These features describe the current layout condition using normalized
    layout-quality scores and selected raw metric indicators.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "layout_score",
        "crossing_score",
        "angular_resolution_score",
        "edge_length_score",
        "node_separation_score",
        "normalized_crossing_count",
        "normalized_local_crossing_number",
        "crossing_edge_fraction",
        "fraction_vertices_with_crossings",
        "edge_length_cv_norm",
        "fraction_nodes_too_close",
        "min_node_distance_ratio_norm",
        "mean_nearest_distance_ratio_norm",
    )

    def __init__(
        self,
        max_edge_length_cv: float = 5.0,
        max_distance_ratio: float = 3.0,
    ):
        self.max_edge_length_cv = max_edge_length_cv
        self.max_distance_ratio = max_distance_ratio

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def extract(
        self,
        context: LayoutContext,
    ) -> FloatDict:
        """
        Extract current layout-quality features from a LayoutContext.
        """
        scores = context.scores
        metrics = context.metrics

        features: FloatDict = {
            "layout_score": clip01(scores.get("layout_score", 0.0)),
            "crossing_score": clip01(scores.get("crossing_score", 0.0)),
            "angular_resolution_score": clip01(
                scores.get("angular_resolution_score", 0.0)
            ),
            "edge_length_score": clip01(scores.get("edge_length_score", 0.0)),
            "node_separation_score": clip01(
                scores.get("node_separation_score", 0.0)
            ),
            "normalized_crossing_count": clip01(
                metrics.get("normalized_crossing_count", 0.0)
            ),
            "normalized_local_crossing_number": clip01(
                metrics.get("normalized_local_crossing_number", 0.0)
            ),
            "crossing_edge_fraction": clip01(
                metrics.get("crossing_edge_fraction", 0.0)
            ),
            "fraction_vertices_with_crossings": clip01(
                metrics.get("fraction_vertices_with_crossings", 0.0)
            ),
            "edge_length_cv_norm": normalize_linear(
                metrics.get("edge_length_cv", 0.0),
                self.max_edge_length_cv,
            ),
            "fraction_nodes_too_close": clip01(
                metrics.get("fraction_nodes_too_close", 0.0)
            ),
            "min_node_distance_ratio_norm": normalize_linear(
                metrics.get("min_node_distance_ratio", 0.0),
                self.max_distance_ratio,
            ),
            "mean_nearest_distance_ratio_norm": normalize_linear(
                metrics.get("mean_nearest_distance_ratio", 0.0),
                self.max_distance_ratio,
            ),
        }

        return self._ordered(features)

    def to_vector(
        self,
        context: LayoutContext,
    ) -> np.ndarray:
        features = self.extract(context)

        return dictionary_to_vector(
            data=features,
            names=self.FEATURE_NAMES,
        )

    def _ordered(
        self,
        features: Dict[str, float],
    ) -> FloatDict:
        return {
            name: safe_float(features.get(name, 0.0))
            for name in self.FEATURE_NAMES
        }