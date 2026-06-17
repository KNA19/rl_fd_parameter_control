from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from envs.layout_context import LayoutContext
from features.normalizers import FloatDict, clip01, dictionary_to_vector, normalize_linear, safe_float


EPSILON = 1e-9


class DynamicsFeatureExtractor:
    """
    Extract layout dynamics and convergence features D_t.

    These features describe whether the layout is expanding, stabilizing,
    moving too much, or becoming stuck.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "progress",
        "layout_width_norm",
        "layout_height_norm",
        "layout_diagonal_norm",
        "layout_area_norm",
        "mean_node_displacement_norm",
        "max_node_displacement_norm",
        "total_node_displacement_norm",
        "mean_iteration_displacement_norm",
        "max_iteration_displacement_norm",
        "displacement_change_rate_norm",
        "temperature_norm_by_layout_scale",
        "temperature_over_mean_displacement_norm",
        "mean_edge_length_over_layout_diagonal",
        "mean_edge_length_over_k_norm",
        "min_node_distance_over_k_norm",
        "mean_nearest_distance_over_k_norm",
    )

    def __init__(
        self,
        max_layout_extent_factor: float = 10.0,
        max_displacement_factor: float = 10.0,
        max_temperature_ratio: float = 20.0,
        max_edge_length_over_k: float = 10.0,
        max_node_distance_over_k: float = 10.0,
    ):
        self.max_layout_extent_factor = max_layout_extent_factor
        self.max_displacement_factor = max_displacement_factor
        self.max_temperature_ratio = max_temperature_ratio
        self.max_edge_length_over_k = max_edge_length_over_k
        self.max_node_distance_over_k = max_node_distance_over_k

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def extract(
        self,
        context: LayoutContext,
    ) -> FloatDict:
        """
        Extract dynamics/convergence features from a LayoutContext.
        """
        layout_scale = max(EPSILON, float(context.layout_scale))

        stats = context.layout_stats
        metrics = context.metrics
        params = context.parameters

        width = safe_float(stats.get("layout_width", 0.0))
        height = safe_float(stats.get("layout_height", 0.0))
        diagonal = safe_float(stats.get("layout_diagonal", 0.0))

        layout_area = width * height
        reference_area = max(EPSILON, (2.0 * layout_scale) ** 2)

        mean_displacement = safe_float(
            stats.get(
                "mean_node_displacement",
                stats.get("context_mean_displacement", 0.0),
            )
        )

        max_displacement = safe_float(
            stats.get(
                "max_node_displacement",
                stats.get("context_max_displacement", 0.0),
            )
        )

        total_displacement = safe_float(
            stats.get(
                "total_node_displacement",
                stats.get("context_total_displacement", 0.0),
            )
        )

        mean_iteration_displacement = safe_float(
            stats.get("mean_iteration_displacement", 0.0)
        )

        max_iteration_displacement = safe_float(
            stats.get("max_iteration_displacement", 0.0)
        )

        previous_mean_displacement = 0.0

        if context.previous_layout_stats is not None:
            previous_mean_displacement = safe_float(
                context.previous_layout_stats.get("mean_node_displacement", 0.0)
            )

        displacement_change_rate = self._ratio_to_unit_interval(
            mean_displacement,
            previous_mean_displacement,
        )

        temperature = safe_float(params.get("temperature", 0.0))
        k_value = max(EPSILON, safe_float(params.get("k", 1.0)))

        mean_edge_length = safe_float(metrics.get("mean_edge_length", 0.0))
        min_node_distance = safe_float(metrics.get("min_node_distance", 0.0))
        mean_nearest_distance = safe_float(
            metrics.get("mean_nearest_neighbor_distance", 0.0)
        )

        features: FloatDict = {
            "progress": clip01(context.progress),
            "layout_width_norm": normalize_linear(
                width,
                self.max_layout_extent_factor * layout_scale,
            ),
            "layout_height_norm": normalize_linear(
                height,
                self.max_layout_extent_factor * layout_scale,
            ),
            "layout_diagonal_norm": normalize_linear(
                diagonal,
                self.max_layout_extent_factor * layout_scale,
            ),
            "layout_area_norm": clip01(layout_area / reference_area),
            "mean_node_displacement_norm": normalize_linear(
                mean_displacement,
                self.max_displacement_factor * layout_scale,
            ),
            "max_node_displacement_norm": normalize_linear(
                max_displacement,
                self.max_displacement_factor * layout_scale,
            ),
            "total_node_displacement_norm": normalize_linear(
                total_displacement,
                self.max_displacement_factor
                * layout_scale
                * max(1, context.graph.number_of_nodes()),
            ),
            "mean_iteration_displacement_norm": normalize_linear(
                mean_iteration_displacement,
                self.max_displacement_factor * layout_scale,
            ),
            "max_iteration_displacement_norm": normalize_linear(
                max_iteration_displacement,
                self.max_displacement_factor * layout_scale,
            ),
            "displacement_change_rate_norm": displacement_change_rate,
            "temperature_norm_by_layout_scale": normalize_linear(
                temperature,
                self.max_displacement_factor * layout_scale,
            ),
            "temperature_over_mean_displacement_norm": normalize_linear(
                temperature / max(EPSILON, mean_displacement),
                self.max_temperature_ratio,
            ),
            "mean_edge_length_over_layout_diagonal": clip01(
                mean_edge_length / max(EPSILON, diagonal)
            ),
            "mean_edge_length_over_k_norm": normalize_linear(
                mean_edge_length / k_value,
                self.max_edge_length_over_k,
            ),
            "min_node_distance_over_k_norm": normalize_linear(
                min_node_distance / k_value,
                self.max_node_distance_over_k,
            ),
            "mean_nearest_distance_over_k_norm": normalize_linear(
                mean_nearest_distance / k_value,
                self.max_node_distance_over_k,
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

    def _ratio_to_unit_interval(
        self,
        current_value: float,
        previous_value: float,
    ) -> float:
        """
        Convert current/previous displacement ratio into [0, 1].

        0.5 means roughly unchanged.
        greater than 0.5 means movement increased.
        less than 0.5 means movement decreased.
        """
        current = max(0.0, safe_float(current_value))
        previous = max(0.0, safe_float(previous_value))

        if current <= EPSILON and previous <= EPSILON:
            return 0.5

        ratio = current / max(EPSILON, previous)
        bounded = ratio / (1.0 + ratio)

        return clip01(bounded)

    def _ordered(
        self,
        features: Dict[str, float],
    ) -> FloatDict:
        return {
            name: safe_float(features.get(name, 0.0))
            for name in self.FEATURE_NAMES
        }