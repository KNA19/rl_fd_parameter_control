from __future__ import annotations

from typing import Mapping

import networkx as nx
import numpy as np

from algorithms.base import (
    BaseForceDirectedAlgorithm,
    LayoutResult,
    LayoutStats,
    Node,
    ParameterDict,
    ParameterSpace,
    PositionDict,
    compute_layout_bounds,
    copy_positions,
)
from algorithms.fruchterman_reingold.defaults import (
    DEFAULT_LAYOUT_SCALE,
    FR_EPSILON,
)
from algorithms.fruchterman_reingold.parameters import (
    FruchtermanReingoldParameterSpace,
)


class FruchtermanReingoldAlgorithm(BaseForceDirectedAlgorithm):
    """
    Pure Fruchterman-Reingold force-directed algorithm.

    Original force equations:

        attractive force = d^2 / k
        repulsive force  = k^2 / d

    The RL agent will later control only:

        k
        temperature
        cooling_rate

    The RL agent will not directly move nodes.
    """

    algorithm_name = "fruchterman_reingold"

    def get_parameter_space(
        self,
        graph: nx.Graph,
        layout_scale: float = DEFAULT_LAYOUT_SCALE,
    ) -> ParameterSpace:
        return FruchtermanReingoldParameterSpace(
            graph=graph,
            layout_scale=layout_scale,
        )

    def default_parameters(
        self,
        graph: nx.Graph,
        layout_scale: float = DEFAULT_LAYOUT_SCALE,
    ) -> ParameterDict:
        return self.get_parameter_space(
            graph=graph,
            layout_scale=layout_scale,
        ).defaults()

    def initialize_layout(
        self,
        graph: nx.Graph,
        seed: int | None = None,
        layout_scale: float = DEFAULT_LAYOUT_SCALE,
    ) -> PositionDict:
        """
        Initialize node positions randomly inside a square layout domain.
        """
        rng = np.random.default_rng(seed)
        scale = float(layout_scale)

        positions: PositionDict = {}

        for node in graph.nodes:
            positions[node] = rng.uniform(
                low=-scale,
                high=scale,
                size=2,
            ).astype(float)

        return positions

    def step(
        self,
        graph: nx.Graph,
        positions: Mapping[Node, np.ndarray],
        parameters: Mapping[str, float],
        iterations: int,
        layout_scale: float = DEFAULT_LAYOUT_SCALE,
    ) -> LayoutResult:
        """
        Run Pure FR for a fixed number of iterations.

        This block-based function is what the RL environment will call after
        each parameter-control action.
        """
        if iterations < 0:
            raise ValueError("iterations must be non-negative.")

        parameter_space = self.get_parameter_space(
            graph=graph,
            layout_scale=layout_scale,
        )

        current_parameters = parameter_space.clip(parameters)
        current_positions = copy_positions(positions)
        start_positions = copy_positions(current_positions)

        nodes = list(graph.nodes)

        if not nodes:
            stats = LayoutStats(iterations=iterations)

            return LayoutResult(
                positions=current_positions,
                parameters=current_parameters,
                stats=stats,
            )

        for node in nodes:
            if node not in current_positions:
                raise KeyError(f"Missing position for graph node: {node}")

        k = float(current_parameters["k"])
        temperature = float(current_parameters["temperature"])
        cooling_rate = float(current_parameters["cooling_rate"])

        initial_temperature = temperature

        mean_iteration_displacements = []
        max_iteration_displacements = []

        for _ in range(iterations):
            displacement: PositionDict = {
                node: np.zeros(2, dtype=float)
                for node in nodes
            }

            self._apply_repulsive_forces(
                nodes=nodes,
                positions=current_positions,
                displacement=displacement,
                k=k,
            )

            self._apply_attractive_forces(
                graph=graph,
                positions=current_positions,
                displacement=displacement,
                k=k,
            )

            moves = []

            for node in nodes:
                disp = displacement[node]
                disp_norm = float(np.linalg.norm(disp))

                if disp_norm <= FR_EPSILON:
                    move = np.zeros(2, dtype=float)
                else:
                    move_length = min(disp_norm, temperature)
                    move = (disp / disp_norm) * move_length

                current_positions[node] = current_positions[node] + move
                moves.append(float(np.linalg.norm(move)))

            if moves:
                mean_iteration_displacements.append(float(np.mean(moves)))
                max_iteration_displacements.append(float(np.max(moves)))

            temperature *= cooling_rate
            temperature = parameter_space.specs["temperature"].clip(
                temperature
            )

        current_parameters["temperature"] = temperature
        current_parameters = parameter_space.clip(current_parameters)

        block_displacements = [
            float(
                np.linalg.norm(
                    current_positions[node] - start_positions[node]
                )
            )
            for node in nodes
        ]

        width, height, diagonal = compute_layout_bounds(current_positions)

        stats = LayoutStats(
            mean_node_displacement=(
                float(np.mean(block_displacements))
                if block_displacements
                else 0.0
            ),
            max_node_displacement=(
                float(np.max(block_displacements))
                if block_displacements
                else 0.0
            ),
            total_node_displacement=(
                float(np.sum(block_displacements))
                if block_displacements
                else 0.0
            ),
            iterations=iterations,
            initial_temperature=initial_temperature,
            final_temperature=float(current_parameters["temperature"]),
            layout_width=width,
            layout_height=height,
            layout_diagonal=diagonal,
            extra={
                "mean_iteration_displacement": (
                    float(np.mean(mean_iteration_displacements))
                    if mean_iteration_displacements
                    else 0.0
                ),
                "max_iteration_displacement": (
                    float(np.max(max_iteration_displacements))
                    if max_iteration_displacements
                    else 0.0
                ),
            },
        )

        return LayoutResult(
            positions=current_positions,
            parameters=current_parameters,
            stats=stats,
        )

    def _apply_repulsive_forces(
        self,
        nodes: list[Node],
        positions: Mapping[Node, np.ndarray],
        displacement: PositionDict,
        k: float,
    ) -> None:
        """
        Apply Pure FR repulsive forces between all node pairs.
        """
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                delta = positions[u] - positions[v]
                distance = float(np.linalg.norm(delta))

                if distance <= FR_EPSILON:
                    delta = np.array([FR_EPSILON, 0.0], dtype=float)
                    distance = FR_EPSILON

                unit = delta / distance
                force = (k * k) / distance
                vector = unit * force

                displacement[u] += vector
                displacement[v] -= vector

    def _apply_attractive_forces(
        self,
        graph: nx.Graph,
        positions: Mapping[Node, np.ndarray],
        displacement: PositionDict,
        k: float,
    ) -> None:
        """
        Apply Pure FR attractive forces along edges.
        """
        for u, v in graph.edges:
            delta = positions[u] - positions[v]
            distance = float(np.linalg.norm(delta))

            if distance <= FR_EPSILON:
                delta = np.array([FR_EPSILON, 0.0], dtype=float)
                distance = FR_EPSILON

            unit = delta / distance
            force = (distance * distance) / max(k, FR_EPSILON)
            vector = unit * force

            displacement[u] -= vector
            displacement[v] += vector