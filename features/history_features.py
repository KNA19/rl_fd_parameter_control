from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from envs.layout_context import LayoutContext
from features.normalizers import FloatDict, clip01, dictionary_to_vector, normalize_linear, safe_float


class HistoryFeatureExtractor:
    """
    Extract metric-trend and action-history features.

    This represents:

        Delta A_t = layout-quality change/trend
        H_t       = recent action history

    The action-dependent feature names are created from the action_names list.
    """

    BASE_FEATURE_NAMES: Tuple[str, ...] = (
        "delta_layout_score_norm",
        "delta_crossing_score_norm",
        "delta_angular_resolution_score_norm",
        "delta_edge_length_score_norm",
        "delta_node_separation_score_norm",
        "delta_normalized_crossing_count_norm",
        "delta_edge_length_cv_norm",
        "delta_fraction_nodes_too_close_norm",
        "same_action_repeat_count_norm",
        "recent_k_action_fraction",
        "recent_temperature_action_fraction",
        "recent_cooling_rate_action_fraction",
        "recent_no_change_fraction",
    )

    def __init__(
        self,
        action_names: List[str],
        history_window: int = 5,
        score_delta_scale: float = 1.0,
        metric_delta_scale: float = 1.0,
    ):
        if not action_names:
            raise ValueError("HistoryFeatureExtractor requires action_names.")

        self.action_names = list(action_names)
        self.history_window = max(1, int(history_window))
        self.score_delta_scale = max(1e-9, float(score_delta_scale))
        self.metric_delta_scale = max(1e-9, float(metric_delta_scale))

        self.last_action_feature_names = tuple(
            f"last_action::{self._sanitize_action_name(action_name)}"
            for action_name in self.action_names
        )

        self.recent_action_feature_names = tuple(
            f"recent_action_fraction::{self._sanitize_action_name(action_name)}"
            for action_name in self.action_names
        )

        self.FEATURE_NAMES: Tuple[str, ...] = (
            self.BASE_FEATURE_NAMES
            + self.last_action_feature_names
            + self.recent_action_feature_names
        )

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def extract(
        self,
        context: LayoutContext,
    ) -> FloatDict:
        features: FloatDict = {
            "delta_layout_score_norm": self._signed_delta_to_unit(
                context.get_score_delta("layout_score", 0.0),
                self.score_delta_scale,
            ),
            "delta_crossing_score_norm": self._signed_delta_to_unit(
                context.get_score_delta("crossing_score", 0.0),
                self.score_delta_scale,
            ),
            "delta_angular_resolution_score_norm": self._signed_delta_to_unit(
                context.get_score_delta("angular_resolution_score", 0.0),
                self.score_delta_scale,
            ),
            "delta_edge_length_score_norm": self._signed_delta_to_unit(
                context.get_score_delta("edge_length_score", 0.0),
                self.score_delta_scale,
            ),
            "delta_node_separation_score_norm": self._signed_delta_to_unit(
                context.get_score_delta("node_separation_score", 0.0),
                self.score_delta_scale,
            ),
            "delta_normalized_crossing_count_norm": self._signed_delta_to_unit(
                context.get_metric_delta("normalized_crossing_count", 0.0),
                self.metric_delta_scale,
            ),
            "delta_edge_length_cv_norm": self._signed_delta_to_unit(
                context.get_metric_delta("edge_length_cv", 0.0),
                self.metric_delta_scale,
            ),
            "delta_fraction_nodes_too_close_norm": self._signed_delta_to_unit(
                context.get_metric_delta("fraction_nodes_too_close", 0.0),
                self.metric_delta_scale,
            ),
            "same_action_repeat_count_norm": clip01(
                context.history.same_action_repeat_count_norm()
            ),
            "recent_k_action_fraction": self._recent_group_fraction(
                context=context,
                keywords=("k",),
            ),
            "recent_temperature_action_fraction": self._recent_group_fraction(
                context=context,
                keywords=("temperature", "temp", "reheat", "cool_down"),
            ),
            "recent_cooling_rate_action_fraction": self._recent_group_fraction(
                context=context,
                keywords=("cooling", "cooling_rate"),
            ),
            "recent_no_change_fraction": self._recent_group_fraction(
                context=context,
                keywords=("no_change",),
            ),
        }

        last_action_vector = context.history.last_action_one_hot(
            num_actions=len(self.action_names)
        )

        for index, action_name in enumerate(self.action_names):
            key = f"last_action::{self._sanitize_action_name(action_name)}"
            features[key] = float(last_action_vector[index])

        for action_name in self.action_names:
            key = f"recent_action_fraction::{self._sanitize_action_name(action_name)}"
            features[key] = clip01(
                context.history.recent_action_fraction(
                    action_name=action_name,
                    window=self.history_window,
                )
            )

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

    def _signed_delta_to_unit(
        self,
        delta: float,
        scale: float,
    ) -> float:
        """
        Map signed delta into [0, 1].

        0.5 = no change
        >0.5 = improvement/increase
        <0.5 = degradation/decrease
        """
        delta_float = safe_float(delta)
        scale_float = max(1e-9, safe_float(scale, default=1.0))

        normalized = 0.5 + 0.5 * (delta_float / scale_float)

        return clip01(normalized)

    def _recent_group_fraction(
        self,
        context: LayoutContext,
        keywords: Tuple[str, ...],
    ) -> float:
        recent_entries = context.history.get_recent_entries(
            window=self.history_window
        )

        if not recent_entries:
            return 0.0

        count = 0

        for entry in recent_entries:
            action_name = entry.action_name.lower()

            if any(keyword.lower() in action_name for keyword in keywords):
                count += 1

        return clip01(count / len(recent_entries))

    def _sanitize_action_name(
        self,
        action_name: str,
    ) -> str:
        return (
            action_name.replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(":", "_")
        )

    def _ordered(
        self,
        features: Dict[str, float],
    ) -> FloatDict:
        return {
            name: safe_float(features.get(name, 0.0))
            for name in self.FEATURE_NAMES
        }