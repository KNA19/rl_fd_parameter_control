from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from envs.layout_context import LayoutContext
from features.normalizers import FloatDict, clip01, dictionary_to_vector, normalize_linear, safe_float


class ConflictFeatureExtractor:
    """
    Extract crossing and local conflict distribution features C_t.

    These features describe whether layout problems are global or concentrated
    around a small number of edges/vertices.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "normalized_crossing_count",
        "normalized_local_crossing_number",
        "crossing_edge_fraction",
        "fraction_vertices_with_crossings",
        "mean_crossings_per_crossed_edge_norm",
        "mean_incident_crossings_per_vertex_norm",
        "max_incident_crossings_per_vertex_norm",
        "std_incident_crossings_per_vertex_norm",
        "top_10_percent_crossing_vertices_ratio",
        "fraction_nodes_too_close",
        "node_closeness_pressure",
    )

    def __init__(
        self,
        max_crossings_per_edge: float = 20.0,
        max_incident_crossings: float = 50.0,
    ):
        self.max_crossings_per_edge = max_crossings_per_edge
        self.max_incident_crossings = max_incident_crossings

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def extract(
        self,
        context: LayoutContext,
    ) -> FloatDict:
        metrics = context.metrics

        min_distance_ratio = safe_float(
            metrics.get("min_node_distance_ratio", 1.0),
            default=1.0,
        )

        node_closeness_pressure = 1.0 - min(1.0, max(0.0, min_distance_ratio))

        features: FloatDict = {
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
            "mean_crossings_per_crossed_edge_norm": normalize_linear(
                metrics.get("mean_crossings_per_crossed_edge", 0.0),
                self.max_crossings_per_edge,
            ),
            "mean_incident_crossings_per_vertex_norm": normalize_linear(
                metrics.get("mean_incident_crossings_per_vertex", 0.0),
                self.max_incident_crossings,
            ),
            "max_incident_crossings_per_vertex_norm": normalize_linear(
                metrics.get("max_incident_crossings_per_vertex", 0.0),
                self.max_incident_crossings,
            ),
            "std_incident_crossings_per_vertex_norm": normalize_linear(
                metrics.get("std_incident_crossings_per_vertex", 0.0),
                self.max_incident_crossings,
            ),
            "top_10_percent_crossing_vertices_ratio": clip01(
                metrics.get("top_10_percent_crossing_vertices_ratio", 0.0)
            ),
            "fraction_nodes_too_close": clip01(
                metrics.get("fraction_nodes_too_close", 0.0)
            ),
            "node_closeness_pressure": clip01(node_closeness_pressure),
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