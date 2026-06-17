from __future__ import annotations

from typing import Dict, Hashable, Mapping

import networkx as nx
import numpy as np


Node = Hashable
PositionDict = Mapping[Node, np.ndarray]
FloatDict = Dict[str, float]

EPSILON = 1e-9


def angle_between_vectors(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Return angle between two vectors in radians.
    """
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))

    if norm_a <= EPSILON or norm_b <= EPSILON:
        return 0.0

    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))

    return float(np.arccos(cosine))


def compute_angular_resolution_metrics(
    graph: nx.Graph,
    positions: PositionDict,
) -> FloatDict:
    """
    Compute angular-resolution metrics.

    For each vertex with degree at least 2, we compute the smallest angle
    between consecutive incident edges. The ideal angle is 2*pi/degree.

    The angular score is the average ratio:

        min_angle / ideal_angle

    clipped to [0, 1].
    """
    min_angles = []
    angle_ratios = []

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))

        if len(neighbors) < 2:
            continue

        if node not in positions:
            continue

        center = np.asarray(positions[node], dtype=float)

        directions = []

        for neighbor in neighbors:
            if neighbor not in positions:
                continue

            vector = np.asarray(positions[neighbor], dtype=float) - center

            if float(np.linalg.norm(vector)) <= EPSILON:
                continue

            angle = float(np.arctan2(vector[1], vector[0]))
            directions.append(angle)

        if len(directions) < 2:
            continue

        directions = sorted(directions)

        angle_differences = []

        for index in range(len(directions)):
            current_angle = directions[index]
            next_angle = directions[(index + 1) % len(directions)]

            if index == len(directions) - 1:
                next_angle += 2.0 * np.pi

            difference = float(next_angle - current_angle)
            angle_differences.append(difference)

        min_angle = float(min(angle_differences))
        ideal_angle = float(2.0 * np.pi / len(directions))

        if ideal_angle <= EPSILON:
            ratio = 1.0
        else:
            ratio = min(1.0, min_angle / ideal_angle)

        min_angles.append(min_angle)
        angle_ratios.append(ratio)

    if not min_angles:
        return {
            "angular_resolution_rad": float(np.pi),
            "angular_resolution_degrees": 180.0,
            "mean_angular_resolution_ratio": 1.0,
            "min_angular_resolution_ratio": 1.0,
        }

    return {
        "angular_resolution_rad": float(np.min(min_angles)),
        "angular_resolution_degrees": float(np.degrees(np.min(min_angles))),
        "mean_angular_resolution_ratio": float(np.mean(angle_ratios)),
        "min_angular_resolution_ratio": float(np.min(angle_ratios)),
    }


def angular_resolution_score_from_metrics(
    metrics: FloatDict,
) -> float:
    """
    Return angular-resolution score in [0, 1].

    Higher is better.
    """
    score = float(metrics.get("mean_angular_resolution_ratio", 1.0))

    return float(max(0.0, min(1.0, score)))