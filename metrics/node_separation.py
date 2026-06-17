from __future__ import annotations

from itertools import combinations
from typing import Dict, Hashable, Mapping

import networkx as nx
import numpy as np


Node = Hashable
PositionDict = Mapping[Node, np.ndarray]
FloatDict = Dict[str, float]

EPSILON = 1e-9


def compute_layout_diagonal(
    positions: PositionDict,
) -> float:
    """
    Compute bounding-box diagonal.
    """
    if not positions:
        return 0.0

    coords = np.asarray(list(positions.values()), dtype=float)

    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))
    y_min = float(np.min(coords[:, 1]))
    y_max = float(np.max(coords[:, 1]))

    width = x_max - x_min
    height = y_max - y_min

    return float(np.sqrt(width * width + height * height))


def compute_node_separation_metrics(
    graph: nx.Graph,
    positions: PositionDict,
) -> FloatDict:
    """
    Compute node-separation and nearest-neighbor metrics.

    A good drawing should avoid node overlap and extreme compression.
    """
    nodes = [node for node in graph.nodes() if node in positions]
    n = len(nodes)

    if n <= 1:
        return {
            "min_node_distance": 1.0,
            "mean_nearest_neighbor_distance": 1.0,
            "std_nearest_neighbor_distance": 0.0,
            "layout_diagonal": 0.0,
            "ideal_node_distance": 1.0,
            "min_node_distance_ratio": 1.0,
            "mean_nearest_distance_ratio": 1.0,
            "fraction_nodes_too_close": 0.0,
        }

    nearest_distances: Dict[Node, float] = {
        node: float("inf") for node in nodes
    }

    all_pair_distances = []

    for u, v in combinations(nodes, 2):
        pos_u = np.asarray(positions[u], dtype=float)
        pos_v = np.asarray(positions[v], dtype=float)

        distance = float(np.linalg.norm(pos_u - pos_v))
        all_pair_distances.append(distance)

        nearest_distances[u] = min(nearest_distances[u], distance)
        nearest_distances[v] = min(nearest_distances[v], distance)

    nearest_values = np.asarray(
        [
            distance
            for distance in nearest_distances.values()
            if np.isfinite(distance)
        ],
        dtype=float,
    )

    if nearest_values.size == 0:
        min_node_distance = 0.0
        mean_nearest = 0.0
        std_nearest = 0.0
    else:
        min_node_distance = float(np.min(nearest_values))
        mean_nearest = float(np.mean(nearest_values))
        std_nearest = float(np.std(nearest_values))

    layout_diagonal = compute_layout_diagonal(positions)

    if layout_diagonal <= EPSILON:
        ideal_node_distance = 1.0
    else:
        ideal_node_distance = layout_diagonal / max(1.0, np.sqrt(float(n)))

    min_ratio = min_node_distance / max(EPSILON, ideal_node_distance)
    mean_ratio = mean_nearest / max(EPSILON, ideal_node_distance)

    too_close_threshold = 0.25 * ideal_node_distance

    if nearest_values.size == 0:
        fraction_too_close = 0.0
    else:
        fraction_too_close = float(
            np.mean(nearest_values < too_close_threshold)
        )

    return {
        "min_node_distance": float(min_node_distance),
        "mean_nearest_neighbor_distance": float(mean_nearest),
        "std_nearest_neighbor_distance": float(std_nearest),
        "layout_diagonal": float(layout_diagonal),
        "ideal_node_distance": float(ideal_node_distance),
        "min_node_distance_ratio": float(min_ratio),
        "mean_nearest_distance_ratio": float(mean_ratio),
        "fraction_nodes_too_close": float(fraction_too_close),
    }


def node_separation_score_from_metrics(
    metrics: FloatDict,
) -> float:
    """
    Convert node-separation quality into a score in [0, 1].

    Higher is better.
    """
    mean_ratio = float(metrics.get("mean_nearest_distance_ratio", 1.0))
    too_close_fraction = float(metrics.get("fraction_nodes_too_close", 0.0))

    spacing_score = min(1.0, mean_ratio)
    penalty = max(0.0, min(1.0, too_close_fraction))

    score = spacing_score * (1.0 - penalty)

    return float(max(0.0, min(1.0, score)))