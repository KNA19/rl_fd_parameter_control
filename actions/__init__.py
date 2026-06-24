from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from actions.action_registry import create_action_space
    from actions.base_action_space import (
        ActionResult,
        ActionSpec,
        BaseParameterActionSpace,
        ParameterOperation,
    )
    from actions.parameter_actions import MultiScaleParameterActionSpace


__all__ = [
    "create_action_space",
    "ActionResult",
    "ActionSpec",
    "BaseParameterActionSpace",
    "ParameterOperation",
    "MultiScaleParameterActionSpace",
]


def __getattr__(name: str) -> Any:
    """
    Lazy imports to avoid circular-import problems.
    """
    if name == "create_action_space":
        from actions.action_registry import create_action_space

        globals()[name] = create_action_space
        return create_action_space

    if name in {
        "ActionResult",
        "ActionSpec",
        "BaseParameterActionSpace",
        "ParameterOperation",
    }:
        from actions import base_action_space

        value = getattr(base_action_space, name)
        globals()[name] = value
        return value

    if name == "MultiScaleParameterActionSpace":
        from actions.parameter_actions import MultiScaleParameterActionSpace

        globals()[name] = MultiScaleParameterActionSpace
        return MultiScaleParameterActionSpace

    raise AttributeError(f"module 'actions' has no attribute '{name}'")