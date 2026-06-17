from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, Mapping, Tuple

import networkx as nx
import numpy as np


Node = Hashable
PositionDict = Dict[Node, np.ndarray]
ParameterDict = Dict[str, float]


@dataclass(frozen=True)
class ParameterSpec:
    """
    Specification for one controllable algorithm parameter.

    This class is algorithm-independent. Pure FR, Eades, Kamada-Kawai,
    or any future force-directed algorithm can define its own parameters
    using this same structure.
    """

    name: str
    default_value: float
    min_value: float
    max_value: float
    scale: str = "linear"
    description: str = ""

    def clip(self, value: float) -> float:
        value_float = float(value)
        return max(self.min_value, min(self.max_value, value_float))

    def normalize(self, value: float) -> float:
        """
        Normalize a parameter value to [0, 1].
        """
        clipped = self.clip(value)

        if self.max_value <= self.min_value:
            return 0.0

        if self.scale == "log":
            eps = 1e-12
            min_value = max(self.min_value, eps)
            max_value = max(self.max_value, min_value + eps)
            clipped = max(clipped, eps)

            normalized = (np.log(clipped) - np.log(min_value)) / (
                np.log(max_value) - np.log(min_value)
            )
        else:
            normalized = (clipped - self.min_value) / (
                self.max_value - self.min_value
            )

        if not np.isfinite(normalized):
            return 0.0

        return float(max(0.0, min(1.0, normalized)))

    def denormalize(self, value: float) -> float:
        """
        Convert a normalized [0, 1] value back to the parameter range.
        """
        normalized = max(0.0, min(1.0, float(value)))

        if self.scale == "log":
            eps = 1e-12
            min_value = max(self.min_value, eps)
            max_value = max(self.max_value, min_value + eps)

            raw_value = np.exp(
                np.log(min_value)
                + normalized * (np.log(max_value) - np.log(min_value))
            )
        else:
            raw_value = self.min_value + normalized * (
                self.max_value - self.min_value
            )

        return self.clip(float(raw_value))


class ParameterSpace:
    """
    General parameter-space container.

    Every force-directed algorithm should define its own ParameterSpace,
    but all parameter spaces should expose the same functions:

        defaults()
        clip()
        normalize()
        denormalize()
    """

    def __init__(self, specs: Iterable[ParameterSpec]):
        spec_tuple = tuple(specs)

        if not spec_tuple:
            raise ValueError("ParameterSpace requires at least one parameter.")

        self.specs: Dict[str, ParameterSpec] = {}

        for spec in spec_tuple:
            if spec.name in self.specs:
                raise ValueError(f"Duplicate parameter name: {spec.name}")

            if spec.min_value > spec.default_value:
                raise ValueError(
                    f"Default value for {spec.name} is below min_value."
                )

            if spec.default_value > spec.max_value:
                raise ValueError(
                    f"Default value for {spec.name} is above max_value."
                )

            self.specs[spec.name] = spec

        self.parameter_names: Tuple[str, ...] = tuple(self.specs.keys())

    def defaults(self) -> ParameterDict:
        return {
            name: spec.default_value
            for name, spec in self.specs.items()
        }

    def clip(self, parameters: Mapping[str, float]) -> ParameterDict:
        clipped: ParameterDict = {}

        for name, spec in self.specs.items():
            value = float(parameters.get(name, spec.default_value))
            clipped[name] = spec.clip(value)

        return clipped

    def normalize(self, parameters: Mapping[str, float]) -> np.ndarray:
        clipped = self.clip(parameters)

        values = [
            self.specs[name].normalize(clipped[name])
            for name in self.parameter_names
        ]

        return np.asarray(values, dtype=np.float32)

    def denormalize(self, values: np.ndarray) -> ParameterDict:
        flat_values = np.asarray(values, dtype=float).reshape(-1)

        if flat_values.shape[0] != len(self.parameter_names):
            raise ValueError(
                f"Expected {len(self.parameter_names)} values, "
                f"got {flat_values.shape[0]}."
            )

        params: ParameterDict = {}

        for index, name in enumerate(self.parameter_names):
            params[name] = self.specs[name].denormalize(
                float(flat_values[index])
            )

        return params

    def as_dict(self) -> Dict[str, Dict[str, float | str]]:
        return {
            name: {
                "default_value": spec.default_value,
                "min_value": spec.min_value,
                "max_value": spec.max_value,
                "scale": spec.scale,
                "description": spec.description,
            }
            for name, spec in self.specs.items()
        }


@dataclass
class LayoutStats:
    """
    Statistics returned after one layout block.

    These values will later be used by the redesigned state space.
    """

    mean_node_displacement: float = 0.0
    max_node_displacement: float = 0.0
    total_node_displacement: float = 0.0
    iterations: int = 0
    initial_temperature: float = 0.0
    final_temperature: float = 0.0
    layout_width: float = 0.0
    layout_height: float = 0.0
    layout_diagonal: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        data: Dict[str, float] = {
            "mean_node_displacement": float(self.mean_node_displacement),
            "max_node_displacement": float(self.max_node_displacement),
            "total_node_displacement": float(self.total_node_displacement),
            "iterations": float(self.iterations),
            "initial_temperature": float(self.initial_temperature),
            "final_temperature": float(self.final_temperature),
            "layout_width": float(self.layout_width),
            "layout_height": float(self.layout_height),
            "layout_diagonal": float(self.layout_diagonal),
        }

        for key, value in self.extra.items():
            data[key] = float(value)

        return data


@dataclass
class LayoutResult:
    """
    Result returned by any force-directed algorithm after running a block.
    """

    positions: PositionDict
    parameters: ParameterDict
    stats: LayoutStats


class BaseForceDirectedAlgorithm(ABC):
    """
    Base interface for all force-directed algorithms.

    Any algorithm added later must implement this interface.

    Examples:
        Fruchterman-Reingold
        Eades
        Kamada-Kawai
        ForceAtlas-style methods
    """

    algorithm_name: str = "base_force_directed_algorithm"

    @abstractmethod
    def get_parameter_space(
        self,
        graph: nx.Graph,
        layout_scale: float = 1.0,
    ) -> ParameterSpace:
        raise NotImplementedError

    @abstractmethod
    def default_parameters(
        self,
        graph: nx.Graph,
        layout_scale: float = 1.0,
    ) -> ParameterDict:
        raise NotImplementedError

    @abstractmethod
    def initialize_layout(
        self,
        graph: nx.Graph,
        seed: int | None = None,
        layout_scale: float = 1.0,
    ) -> PositionDict:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        graph: nx.Graph,
        positions: Mapping[Node, np.ndarray],
        parameters: Mapping[str, float],
        iterations: int,
        layout_scale: float = 1.0,
    ) -> LayoutResult:
        raise NotImplementedError

    def run(
        self,
        graph: nx.Graph,
        positions: Mapping[Node, np.ndarray] | None = None,
        parameters: Mapping[str, float] | None = None,
        total_iterations: int = 200,
        layout_scale: float = 1.0,
        seed: int | None = None,
    ) -> LayoutResult:
        """
        Run the algorithm for total_iterations.

        The RL environment will mostly use step(), but run() is useful
        for baselines and testing.
        """
        if positions is None:
            current_positions = self.initialize_layout(
                graph=graph,
                seed=seed,
                layout_scale=layout_scale,
            )
        else:
            current_positions = copy_positions(positions)

        if parameters is None:
            current_parameters = self.default_parameters(
                graph=graph,
                layout_scale=layout_scale,
            )
        else:
            current_parameters = dict(parameters)

        return self.step(
            graph=graph,
            positions=current_positions,
            parameters=current_parameters,
            iterations=total_iterations,
            layout_scale=layout_scale,
        )


def copy_positions(
    positions: Mapping[Node, np.ndarray],
) -> PositionDict:
    """
    Deep-copy a node-position dictionary.
    """
    return {
        node: np.asarray(coord, dtype=float).copy()
        for node, coord in positions.items()
    }


def compute_layout_bounds(
    positions: Mapping[Node, np.ndarray],
) -> Tuple[float, float, float]:
    """
    Compute width, height, and diagonal of the current layout.
    """
    if not positions:
        return 0.0, 0.0, 0.0

    coords = np.asarray(list(positions.values()), dtype=float)

    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))
    y_min = float(np.min(coords[:, 1]))
    y_max = float(np.max(coords[:, 1]))

    width = x_max - x_min
    height = y_max - y_min
    diagonal = float(np.sqrt(width * width + height * height))

    return width, height, diagonal