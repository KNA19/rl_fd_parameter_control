from __future__ import annotations

from typing import List

from algorithms.base import ParameterSpace
from states.state_builder import StateBuilder


def create_state_builder(
    state_name: str,
    parameter_space: ParameterSpace,
    action_names: List[str],
) -> StateBuilder:
    """
    Create a StateBuilder by name.

    This registry supports future ablation studies.
    """
    normalized_name = state_name.lower().strip()

    if normalized_name in ("full", "state_v2", "publishable_v2"):
        return StateBuilder(
            parameter_space=parameter_space,
            action_names=action_names,
            include_graph_features=True,
            include_graph_embedding=True,
            include_parameter_features=True,
            include_layout_features=True,
            include_dynamics_features=True,
            include_conflict_features=True,
            include_history_features=True,
        )

    if normalized_name == "no_graph_embedding":
        return StateBuilder(
            parameter_space=parameter_space,
            action_names=action_names,
            include_graph_features=True,
            include_graph_embedding=False,
            include_parameter_features=True,
            include_layout_features=True,
            include_dynamics_features=True,
            include_conflict_features=True,
            include_history_features=True,
        )

    if normalized_name == "no_history":
        return StateBuilder(
            parameter_space=parameter_space,
            action_names=action_names,
            include_graph_features=True,
            include_graph_embedding=True,
            include_parameter_features=True,
            include_layout_features=True,
            include_dynamics_features=True,
            include_conflict_features=True,
            include_history_features=False,
        )

    if normalized_name == "no_conflict":
        return StateBuilder(
            parameter_space=parameter_space,
            action_names=action_names,
            include_graph_features=True,
            include_graph_embedding=True,
            include_parameter_features=True,
            include_layout_features=True,
            include_dynamics_features=True,
            include_conflict_features=False,
            include_history_features=True,
        )

    if normalized_name == "no_dynamics":
        return StateBuilder(
            parameter_space=parameter_space,
            action_names=action_names,
            include_graph_features=True,
            include_graph_embedding=True,
            include_parameter_features=True,
            include_layout_features=True,
            include_dynamics_features=False,
            include_conflict_features=True,
            include_history_features=True,
        )

    if normalized_name == "minimal":
        return StateBuilder(
            parameter_space=parameter_space,
            action_names=action_names,
            include_graph_features=True,
            include_graph_embedding=False,
            include_parameter_features=True,
            include_layout_features=True,
            include_dynamics_features=False,
            include_conflict_features=False,
            include_history_features=False,
        )

    available = [
        "full",
        "state_v2",
        "publishable_v2",
        "no_graph_embedding",
        "no_history",
        "no_conflict",
        "no_dynamics",
        "minimal",
    ]

    raise ValueError(
        f"Unknown state builder: {state_name}. "
        f"Available options: {available}"
    )