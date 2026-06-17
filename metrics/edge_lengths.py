from __future__ import annotations

from typing import Dict, Hashable, Mapping

import networkx as nx
import numpy as np


Node = Hashable
PositionDict = Mapping[Node, np.ndarray]
FloatDict = Dict[str, float]

EPSILON = 1e-9


def compute_edge_length_metrics(
    graph: nx.Graph,
    positions: PositionDict,
) -> FloatDict:
    """
    Compute edge-length statistics.

    Main value for layout quality:

        edge_length_cv = std(edge_lengths) / mean(edge_lengths)

    Lower coefficient of variation is better.
    """
    lengths = []

    for u, v in graph.edges():
        if u not in positions or v not in positions:
            continue

        pos_u = np.asarray(positions[u], dtype=float)
        pos_v = np.asarray(positions[v], dtype=float)

        length = float(np.linalg.norm(pos_u - pos_v))
        lengths.append(length)

    if not lengths:
        return {
            "mean_edge_length": 0.0,
            "std_edge_length": 0.0,
            "min_edge_length": 0.0,
            "max_edge_length": 0.0,
            "edge_length_cv": 0.0,
            "edge_length_variation": 0.0,
        }

    length_array = np.asarray(lengths, dtype=float)

    mean_length = float(np.mean(length_array))
    std_length = float(np.std(length_array))
    min_length = float(np.min(length_array))
    max_length = float(np.max(length_array))

    if mean_length <= EPSILON:
        edge_length_cv = 0.0
    else:
        edge_length_cv = std_length / mean_length

    return {
        "mean_edge_length": float(mean_length),
        "std_edge_length": float(std_length),
        "min_edge_length": float(min_length),
        "max_edge_length": float(max_length),
        "edge_length_cv": float(edge_length_cv),
        "edge_length_variation": float(edge_length_cv),
    }


def edge_length_score_from_metrics(
    metrics: FloatDict,
) -> float:
    """
    Convert edge-length variation into a score in [0, 1].

    Higher is better. A coefficient of variation near 0 gives score near 1.
    """
    cv = float(metrics.get("edge_length_cv", 0.0))
    score = 1.0 / (1.0 + cv)

    return float(max(0.0, min(1.0, score)))