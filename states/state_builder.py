from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from gymnasium import spaces

from algorithms.base import ParameterSpace
from envs.layout_context import LayoutContext
from features import (
    ConflictFeatureExtractor,
    DynamicsFeatureExtractor,
    GraphFeatureExtractor,
    HistoryFeatureExtractor,
    LayoutFeatureExtractor,
    SpectralGraphEmbedding,
)
from features.normalizers import FloatDict, clip01, dictionary_to_vector, safe_float
from states.state_schema import StateSchema


EPSILON = 1e-9


class ParameterStateFeatureExtractor:
    """
    Extract algorithm-parameter features P_t.

    This extractor is algorithm-general. It uses ParameterSpace rather than
    hardcoding Pure FR parameters.

    For each parameter, it produces:

        parameter_norm::<name>
        parameter_default_ratio::<name>
        parameter_delta::<name>

    The meaning is:

        parameter_norm:
            current value normalized to [0, 1] using ParameterSpec.

        parameter_default_ratio:
            current/default ratio mapped to [0, 1].
            0.5 approximately means current equals default.

        parameter_delta:
            current - previous parameter value, normalized to [0, 1].
            0.5 means no change.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
    ):
        self.parameter_names: Tuple[str, ...] = tuple(
            parameter_space.parameter_names
        )

        names: List[str] = []

        for parameter_name in self.parameter_names:
            names.append(f"parameter_norm::{parameter_name}")

        for parameter_name in self.parameter_names:
            names.append(f"parameter_default_ratio::{parameter_name}")

        for parameter_name in self.parameter_names:
            names.append(f"parameter_delta::{parameter_name}")

        self.FEATURE_NAMES: Tuple[str, ...] = tuple(names)

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def extract(
        self,
        context: LayoutContext,
        parameter_space: ParameterSpace,
    ) -> FloatDict:
        """
        Extract parameter-state features from the current context.
        """
        features: FloatDict = {}

        normalized_values = parameter_space.normalize(context.parameters)

        for index, parameter_name in enumerate(self.parameter_names):
            value = 0.0

            if index < normalized_values.shape[0]:
                value = float(normalized_values[index])

            features[f"parameter_norm::{parameter_name}"] = clip01(value)

        for parameter_name in self.parameter_names:
            spec = parameter_space.specs[parameter_name]

            current_value = safe_float(
                context.parameters.get(parameter_name, spec.default_value)
            )

            default_value = max(EPSILON, safe_float(spec.default_value))

            ratio = current_value / default_value
            ratio_unit = ratio / (1.0 + ratio)

            features[f"parameter_default_ratio::{parameter_name}"] = clip01(
                ratio_unit
            )

        for parameter_name in self.parameter_names:
            spec = parameter_space.specs[parameter_name]

            parameter_range = max(
                EPSILON,
                safe_float(spec.max_value) - safe_float(spec.min_value),
            )

            delta = context.get_parameter_delta(
                parameter_name=parameter_name,
                default=0.0,
            )

            delta_unit = 0.5 + 0.5 * (delta / parameter_range)

            features[f"parameter_delta::{parameter_name}"] = clip01(delta_unit)

        return self._ordered(features)

    def to_vector(
        self,
        context: LayoutContext,
        parameter_space: ParameterSpace,
    ) -> np.ndarray:
        features = self.extract(
            context=context,
            parameter_space=parameter_space,
        )

        return dictionary_to_vector(
            data=features,
            names=self.FEATURE_NAMES,
        )

    def _ordered(
        self,
        features: Mapping[str, float],
    ) -> FloatDict:
        return {
            name: safe_float(features.get(name, 0.0))
            for name in self.FEATURE_NAMES
        }


class StateBuilder:
    """
    Final redesigned state builder.

    It constructs:

        s_t = [
            G_h,
            G_e,
            P_t,
            A_t,
            D_t,
            C_t,
            H_t
        ]

    where:

        G_h = handcrafted graph descriptors
        G_e = spectral graph embedding
        P_t = algorithm parameter state
        A_t = current layout-quality features
        D_t = layout dynamics/convergence features
        C_t = conflict/crossing distribution features
        H_t = history and metric-delta features

    Note:
        ΔA_t is included inside HistoryFeatureExtractor.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        action_names: List[str],
        include_graph_features: bool = True,
        include_graph_embedding: bool = True,
        include_parameter_features: bool = True,
        include_layout_features: bool = True,
        include_dynamics_features: bool = True,
        include_conflict_features: bool = True,
        include_history_features: bool = True,
    ):
        if not action_names:
            raise ValueError("StateBuilder requires at least one action name.")

        self.parameter_space = parameter_space
        self.action_names = list(action_names)

        self.include_graph_features = include_graph_features
        self.include_graph_embedding = include_graph_embedding
        self.include_parameter_features = include_parameter_features
        self.include_layout_features = include_layout_features
        self.include_dynamics_features = include_dynamics_features
        self.include_conflict_features = include_conflict_features
        self.include_history_features = include_history_features

        self.graph_feature_extractor = GraphFeatureExtractor()
        self.graph_embedding_extractor = SpectralGraphEmbedding()
        self.parameter_feature_extractor = ParameterStateFeatureExtractor(
            parameter_space=parameter_space
        )
        self.layout_feature_extractor = LayoutFeatureExtractor()
        self.dynamics_feature_extractor = DynamicsFeatureExtractor()
        self.conflict_feature_extractor = ConflictFeatureExtractor()
        self.history_feature_extractor = HistoryFeatureExtractor(
            action_names=self.action_names,
            history_window=5,
        )

        self.schema = self._build_schema()

    @property
    def observation_dim(self) -> int:
        return self.schema.observation_dim

    def build(
        self,
        context: LayoutContext,
        parameter_space: Optional[ParameterSpace] = None,
    ) -> np.ndarray:
        """
        Build the final observation vector.

        All returned values are finite float32 values clipped to [0, 1].
        """
        active_parameter_space = (
            parameter_space if parameter_space is not None else self.parameter_space
        )

        self._validate_parameter_space(active_parameter_space)

        component_vectors = self.build_component_vectors(
            context=context,
            parameter_space=active_parameter_space,
        )

        vectors = [
            component_vectors[component.name]
            for component in self.schema.components
        ]

        if not vectors:
            return np.zeros(0).astype("float32")

        observation = np.concatenate(vectors).astype("float32")
        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).astype("float32")

        observation = np.clip(observation, 0.0, 1.0).astype("float32")

        if observation.shape[0] != self.observation_dim:
            raise ValueError(
                f"State dimension mismatch. Expected {self.observation_dim}, "
                f"got {observation.shape[0]}."
            )

        return observation

    def build_component_vectors(
        self,
        context: LayoutContext,
        parameter_space: Optional[ParameterSpace] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return state vectors by component.

        Useful for debugging and ablation analysis.
        """
        active_parameter_space = (
            parameter_space if parameter_space is not None else self.parameter_space
        )

        self._validate_parameter_space(active_parameter_space)

        vectors: Dict[str, np.ndarray] = {}

        if self.include_graph_features:
            vectors["graph_features"] = self.graph_feature_extractor.to_vector(
                context.graph
            )

        if self.include_graph_embedding:
            vectors["graph_embedding"] = (
                self.graph_embedding_extractor.to_vector(context.graph)
            )

        if self.include_parameter_features:
            vectors["parameter_features"] = (
                self.parameter_feature_extractor.to_vector(
                    context=context,
                    parameter_space=active_parameter_space,
                )
            )

        if self.include_layout_features:
            vectors["layout_features"] = self.layout_feature_extractor.to_vector(
                context
            )

        if self.include_dynamics_features:
            vectors["dynamics_features"] = (
                self.dynamics_feature_extractor.to_vector(context)
            )

        if self.include_conflict_features:
            vectors["conflict_features"] = (
                self.conflict_feature_extractor.to_vector(context)
            )

        if self.include_history_features:
            vectors["history_features"] = (
                self.history_feature_extractor.to_vector(context)
            )

        return vectors

    def make_observation_space(self) -> spaces.Box:
        """
        Create Gymnasium observation space.
        """
        low = np.zeros(self.observation_dim).astype("float32")
        high = np.ones(self.observation_dim).astype("float32")

        return spaces.Box(
            low=low,
            high=high,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

    def component_dimensions(self) -> Dict[str, int]:
        return self.schema.component_dimensions()

    def print_state_definition(self) -> None:
        self.schema.print_summary()

    def _build_schema(self) -> StateSchema:
        component_features: Dict[str, Tuple[str, ...]] = {}

        if self.include_graph_features:
            component_features["graph_features"] = (
                self.graph_feature_extractor.FEATURE_NAMES
            )

        if self.include_graph_embedding:
            component_features["graph_embedding"] = (
                self.graph_embedding_extractor.EMBEDDING_NAMES
            )

        if self.include_parameter_features:
            component_features["parameter_features"] = (
                self.parameter_feature_extractor.FEATURE_NAMES
            )

        if self.include_layout_features:
            component_features["layout_features"] = (
                self.layout_feature_extractor.FEATURE_NAMES
            )

        if self.include_dynamics_features:
            component_features["dynamics_features"] = (
                self.dynamics_feature_extractor.FEATURE_NAMES
            )

        if self.include_conflict_features:
            component_features["conflict_features"] = (
                self.conflict_feature_extractor.FEATURE_NAMES
            )

        if self.include_history_features:
            component_features["history_features"] = (
                self.history_feature_extractor.FEATURE_NAMES
            )

        return StateSchema.from_components(component_features)

    def _validate_parameter_space(
        self,
        parameter_space: ParameterSpace,
    ) -> None:
        current_names = tuple(parameter_space.parameter_names)

        if current_names != self.parameter_feature_extractor.parameter_names:
            raise ValueError(
                "Parameter-space mismatch. "
                f"Expected {self.parameter_feature_extractor.parameter_names}, "
                f"got {current_names}."
            )