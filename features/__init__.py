from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from features.conflict_features import ConflictFeatureExtractor
    from features.dynamics_features import DynamicsFeatureExtractor
    from features.graph_embedding import SpectralGraphEmbedding
    from features.graph_features import GraphFeatureExtractor
    from features.history_features import HistoryFeatureExtractor
    from features.layout_features import LayoutFeatureExtractor
    from features.normalizers import (
        FloatDict,
        clean_feature_dict,
        clip01,
        dictionary_to_vector,
        normalize_linear,
        normalize_log1p,
        normalize_signed_unit,
        safe_float,
    )


__all__ = [
    "GraphFeatureExtractor",
    "SpectralGraphEmbedding",
    "LayoutFeatureExtractor",
    "DynamicsFeatureExtractor",
    "ConflictFeatureExtractor",
    "HistoryFeatureExtractor",
    "FloatDict",
    "clean_feature_dict",
    "clip01",
    "dictionary_to_vector",
    "normalize_linear",
    "normalize_log1p",
    "normalize_signed_unit",
    "safe_float",
]


def __getattr__(name: str) -> Any:
    """
    Lazy imports to avoid circular-import problems between features, envs,
    states, and actions.
    """
    if name == "GraphFeatureExtractor":
        from features.graph_features import GraphFeatureExtractor

        globals()[name] = GraphFeatureExtractor
        return GraphFeatureExtractor

    if name == "SpectralGraphEmbedding":
        from features.graph_embedding import SpectralGraphEmbedding

        globals()[name] = SpectralGraphEmbedding
        return SpectralGraphEmbedding

    if name == "LayoutFeatureExtractor":
        from features.layout_features import LayoutFeatureExtractor

        globals()[name] = LayoutFeatureExtractor
        return LayoutFeatureExtractor

    if name == "DynamicsFeatureExtractor":
        from features.dynamics_features import DynamicsFeatureExtractor

        globals()[name] = DynamicsFeatureExtractor
        return DynamicsFeatureExtractor

    if name == "ConflictFeatureExtractor":
        from features.conflict_features import ConflictFeatureExtractor

        globals()[name] = ConflictFeatureExtractor
        return ConflictFeatureExtractor

    if name == "HistoryFeatureExtractor":
        from features.history_features import HistoryFeatureExtractor

        globals()[name] = HistoryFeatureExtractor
        return HistoryFeatureExtractor

    if name in {
        "FloatDict",
        "clean_feature_dict",
        "clip01",
        "dictionary_to_vector",
        "normalize_linear",
        "normalize_log1p",
        "normalize_signed_unit",
        "safe_float",
    }:
        from features import normalizers

        value = getattr(normalizers, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'features' has no attribute '{name}'")