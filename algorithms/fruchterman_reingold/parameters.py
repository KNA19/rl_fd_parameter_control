from __future__ import annotations

import math

import networkx as nx

from algorithms.base import ParameterSpace, ParameterSpec
from algorithms.fruchterman_reingold.defaults import (
    COOLING_RATE_MAX,
    COOLING_RATE_MIN,
    DEFAULT_COOLING_RATE,
    DEFAULT_LAYOUT_SCALE,
    DEFAULT_TEMPERATURE_FRACTION,
    K_MAX_FACTOR,
    K_MIN_FACTOR,
    MIN_POSITIVE_VALUE,
    TEMPERATURE_MAX_FACTOR,
    TEMPERATURE_MIN_FACTOR,
)


def estimate_default_k(
    graph: nx.Graph,
    layout_scale: float = DEFAULT_LAYOUT_SCALE,
) -> float:
    """
    Estimate the default Pure FR ideal distance.

    Classic FR uses:

        k = sqrt(area / number_of_nodes)

    Here the layout domain is approximately:
        [-layout_scale, layout_scale] x [-layout_scale, layout_scale]
    """
    n = max(1, graph.number_of_nodes())

    width = 2.0 * float(layout_scale)
    height = 2.0 * float(layout_scale)
    area = max(MIN_POSITIVE_VALUE, width * height)

    return float(math.sqrt(area / n))


def estimate_default_temperature(
    layout_scale: float = DEFAULT_LAYOUT_SCALE,
) -> float:
    """
    Estimate default starting temperature.
    """
    drawing_width = 2.0 * float(layout_scale)

    return max(
        MIN_POSITIVE_VALUE,
        DEFAULT_TEMPERATURE_FRACTION * drawing_width,
    )


class FruchtermanReingoldParameterSpace(ParameterSpace):
    """
    Parameter space for Pure Fruchterman-Reingold.

    Controllable parameters:

        k
        temperature
        cooling_rate

    No gravity, no attraction multiplier, and no repulsion multiplier are used,
    because this is the Pure FR version.
    """

    def __init__(
        self,
        graph: nx.Graph,
        layout_scale: float = DEFAULT_LAYOUT_SCALE,
    ):
        default_k = estimate_default_k(
            graph=graph,
            layout_scale=layout_scale,
        )

        default_temperature = estimate_default_temperature(
            layout_scale=layout_scale,
        )

        specs = (
            ParameterSpec(
                name="k",
                default_value=default_k,
                min_value=max(MIN_POSITIVE_VALUE, K_MIN_FACTOR * default_k),
                max_value=max(MIN_POSITIVE_VALUE, K_MAX_FACTOR * default_k),
                scale="log",
                description="Ideal node distance in Pure Fruchterman-Reingold.",
            ),
            ParameterSpec(
                name="temperature",
                default_value=default_temperature,
                min_value=max(
                    MIN_POSITIVE_VALUE,
                    TEMPERATURE_MIN_FACTOR * layout_scale,
                ),
                max_value=max(
                    MIN_POSITIVE_VALUE,
                    TEMPERATURE_MAX_FACTOR * layout_scale,
                ),
                scale="log",
                description="Maximum node displacement per FR iteration.",
            ),
            ParameterSpec(
                name="cooling_rate",
                default_value=DEFAULT_COOLING_RATE,
                min_value=COOLING_RATE_MIN,
                max_value=COOLING_RATE_MAX,
                scale="linear",
                description="Multiplicative temperature cooling factor.",
            ),
        )

        super().__init__(specs=specs)