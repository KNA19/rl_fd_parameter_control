from __future__ import annotations

from algorithms.base import BaseForceDirectedAlgorithm
from algorithms.fruchterman_reingold import FruchtermanReingoldAlgorithm
from envs.fd_param_control_env import (
    FDParamControlEnvConfig,
    ForceDirectedParameterControlEnv,
)


def create_algorithm(
    algorithm_name: str,
) -> BaseForceDirectedAlgorithm:
    """
    Create a force-directed algorithm by name.

    For now, Pure Fruchterman-Reingold is implemented.
    More algorithms can be added later using the same interface.
    """
    normalized_name = algorithm_name.lower().strip()

    if normalized_name in {
        "fruchterman_reingold",
        "pure_fr",
        "fr",
        "default",
    }:
        return FruchtermanReingoldAlgorithm()

    available = [
        "fruchterman_reingold",
        "pure_fr",
        "fr",
        "default",
    ]

    raise ValueError(
        f"Unknown algorithm: {algorithm_name}. "
        f"Available algorithms: {available}"
    )


def create_env(
    config: FDParamControlEnvConfig,
    algorithm_name: str = "fruchterman_reingold",
) -> ForceDirectedParameterControlEnv:
    """
    Create the final SARL force-directed parameter-control environment.
    """
    algorithm = create_algorithm(algorithm_name)

    return ForceDirectedParameterControlEnv(
        config=config,
        algorithm=algorithm,
    )