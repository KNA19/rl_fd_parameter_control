from __future__ import annotations

from algorithms.base import ParameterSpace
from actions.parameter_actions import MultiScaleParameterActionSpace


def create_action_space(
    action_space_name: str,
    parameter_space: ParameterSpace,
    algorithm_name: str = "generic",
) -> MultiScaleParameterActionSpace:
    """
    Create an action space by name.

    This registry supports future algorithms and ablation studies.
    """
    normalized_name = action_space_name.lower().strip()

    if normalized_name in {
        "multiscale",
        "parameter_multiscale",
        "pure_fr_multiscale",
        "default",
    }:
        return MultiScaleParameterActionSpace(
            parameter_space=parameter_space,
            algorithm_name=algorithm_name,
        )

    available = [
        "multiscale",
        "parameter_multiscale",
        "pure_fr_multiscale",
        "default",
    ]

    raise ValueError(
        f"Unknown action space: {action_space_name}. "
        f"Available options: {available}"
    )