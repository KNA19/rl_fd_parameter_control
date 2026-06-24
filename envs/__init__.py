from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from envs.env_factory import create_algorithm, create_env
    from envs.fd_param_control_env import (
        FDParamControlEnvConfig,
        ForceDirectedParameterControlEnv,
    )
    from envs.layout_context import LayoutContext


__all__ = [
    "LayoutContext",
    "FDParamControlEnvConfig",
    "ForceDirectedParameterControlEnv",
    "create_algorithm",
    "create_env",
]


def __getattr__(name: str) -> Any:
    """
    Lazy imports to avoid circular imports between envs, actions, states,
    and features.
    """
    if name == "LayoutContext":
        from envs.layout_context import LayoutContext

        globals()[name] = LayoutContext
        return LayoutContext

    if name in {"create_algorithm", "create_env"}:
        from envs import env_factory

        value = getattr(env_factory, name)
        globals()[name] = value
        return value

    if name in {"FDParamControlEnvConfig", "ForceDirectedParameterControlEnv"}:
        from envs import fd_param_control_env

        value = getattr(fd_param_control_env, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'envs' has no attribute '{name}'")