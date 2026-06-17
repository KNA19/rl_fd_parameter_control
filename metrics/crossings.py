from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Hashable, Mapping, Tuple

import networkx as nx
import numpy as np


Node = Hashable
PositionDict = Mapping[Node, np.ndarray]
EdgeKey = Tuple[Node, Node]
FloatDict = Dict[str, float]

EPSILON = 1e-9


def normalized_edge(u: Node, v: Node) -> EdgeKey:
    """
    Return a stable undirected edge key.
    """
    return tuple(sorted((u, v), key=lambda x: str(x)))  # type: ignore[return-value]


def orientation(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """
    Signed area / orientation test.

    Positive: counter-clockwise
    Negative: clockwise
    Zero: collinear
    """
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def on_segment(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    """
    Return True if point b lies on segment ac.
    """
    return (
        min(a[0], c[0]) - EPSILON <= b[0] <= max(a[0], c[0]) + EPSILON
        and min(a[1], c[1]) - EPSILON <= b[1] <= max(a[1], c[1]) + EPSILON
    )


def segments_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> bool:
    """
    Check whether two straight-line segments intersect.

    This function is used only for non-adjacent graph edges, so graph-edge
    endpoint sharing is already removed before this function is called.

    Collinear overlap is counted as an intersection because it is visually
    problematic in a straight-line drawing.
    """
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # Proper intersection.
    if (o1 * o2 < -EPSILON) and (o3 * o4 < -EPSILON):
        return True

    # Collinear / touching cases.
    if abs(o1) <= EPSILON and on_segment(p1, q1, p2):
        return True

    if abs(o2) <= EPSILON and on_segment(p1, q2, p2):
        return True

    if abs(o3) <= EPSILON and on_segment(q1, p1, q2):
        return True

    if abs(o4) <= EPSILON and on_segment(q1, p2, q2):
        return True

    return False


def edges_share_endpoint(
    edge_a: EdgeKey,
    edge_b: EdgeKey,
) -> bool:
    """
    Return True if two graph edges share an endpoint.
    """
    return bool(set(edge_a).intersection(set(edge_b)))


def compute_purchase_crossing_upper_bound(
    graph: nx.Graph,
) -> float:
    """
    Compute the Purchase-style upper bound for possible edge crossings.

    This excludes crossings between adjacent edges because adjacent edges
    share a node and should not be counted as valid edge crossings.

    Formula:
        chi_star =
            m(m - 1) / 2
            - 1/2 * sum_v deg(v)(deg(v) - 1)
    """
    m = graph.number_of_edges()

    total_edge_pairs = m * (m - 1) / 2.0

    adjacent_edge_pairs = 0.0

    for _node, degree in graph.degree():
        adjacent_edge_pairs += degree * (degree - 1) / 2.0

    upper_bound = total_edge_pairs - adjacent_edge_pairs

    return float(max(0.0, upper_bound))


def compute_crossing_metrics(
    graph: nx.Graph,
    positions: PositionDict,
) -> FloatDict:
    """
    Compute global and local crossing metrics for a straight-line drawing.

    Returned values include:

        crossing_count
        normalized_crossing_count
        local_crossing_number
        normalized_local_crossing_number
        crossing_edge_fraction
        fraction_vertices_with_crossings
        mean_crossings_per_crossed_edge
        mean_incident_crossings_per_vertex
        max_incident_crossings_per_vertex
        std_incident_crossings_per_vertex
        top_10_percent_crossing_vertices_ratio
    """
    edges = [normalized_edge(u, v) for u, v in graph.edges()]
    m = len(edges)
    n = graph.number_of_nodes()

    edge_crossing_counts: Dict[EdgeKey, int] = {edge: 0 for edge in edges}
    vertex_incident_crossings: Dict[Node, int] = {
        node: 0 for node in graph.nodes()
    }

    crossing_count = 0

    for edge_a, edge_b in combinations(edges, 2):
        if edges_share_endpoint(edge_a, edge_b):
            continue

        u1, v1 = edge_a
        u2, v2 = edge_b

        if (
            u1 not in positions
            or v1 not in positions
            or u2 not in positions
            or v2 not in positions
        ):
            continue

        p1 = np.asarray(positions[u1], dtype=float)
        p2 = np.asarray(positions[v1], dtype=float)
        q1 = np.asarray(positions[u2], dtype=float)
        q2 = np.asarray(positions[v2], dtype=float)

        if segments_intersect(p1, p2, q1, q2):
            crossing_count += 1

            edge_crossing_counts[edge_a] += 1
            edge_crossing_counts[edge_b] += 1

            vertex_incident_crossings[u1] += 1
            vertex_incident_crossings[v1] += 1
            vertex_incident_crossings[u2] += 1
            vertex_incident_crossings[v2] += 1

    crossing_upper_bound = compute_purchase_crossing_upper_bound(graph)

    if crossing_upper_bound <= EPSILON:
        normalized_crossing_count = 0.0
    else:
        normalized_crossing_count = crossing_count / crossing_upper_bound

    normalized_crossing_count = float(
        max(0.0, min(1.0, normalized_crossing_count))
    )

    local_crossing_number = (
        max(edge_crossing_counts.values()) if edge_crossing_counts else 0
    )

    normalized_local_crossing_number = local_crossing_number / max(1, m)

    crossed_edges = [
        edge for edge, count in edge_crossing_counts.items() if count > 0
    ]

    crossing_edge_fraction = len(crossed_edges) / max(1, m)

    vertices_with_crossings = [
        node
        for node, count in vertex_incident_crossings.items()
        if count > 0
    ]

    fraction_vertices_with_crossings = len(vertices_with_crossings) / max(1, n)

    if crossed_edges:
        mean_crossings_per_crossed_edge = float(
            np.mean([edge_crossing_counts[edge] for edge in crossed_edges])
        )
    else:
        mean_crossings_per_crossed_edge = 0.0

    incident_values = np.asarray(
        list(vertex_incident_crossings.values()),
        dtype=float,
    )

    if incident_values.size == 0:
        mean_incident = 0.0
        max_incident = 0.0
        std_incident = 0.0
        top_10_ratio = 0.0
    else:
        mean_incident = float(np.mean(incident_values))
        max_incident = float(np.max(incident_values))
        std_incident = float(np.std(incident_values))

        sorted_values = np.sort(incident_values)[::-1]
        top_k = max(1, int(np.ceil(0.10 * len(sorted_values))))
        total_incident = float(np.sum(sorted_values))

        if total_incident <= EPSILON:
            top_10_ratio = 0.0
        else:
            top_10_ratio = float(np.sum(sorted_values[:top_k]) / total_incident)

    return {
        "crossing_count": float(crossing_count),
        "crossing_upper_bound": float(crossing_upper_bound),
        "normalized_crossing_count": float(normalized_crossing_count),
        "local_crossing_number": float(local_crossing_number),
        "normalized_local_crossing_number": float(
            normalized_local_crossing_number
        ),
        "crossing_edge_fraction": float(crossing_edge_fraction),
        "fraction_vertices_with_crossings": float(
            fraction_vertices_with_crossings
        ),
        "mean_crossings_per_crossed_edge": float(
            mean_crossings_per_crossed_edge
        ),
        "mean_incident_crossings_per_vertex": float(mean_incident),
        "max_incident_crossings_per_vertex": float(max_incident),
        "std_incident_crossings_per_vertex": float(std_incident),
        "top_10_percent_crossing_vertices_ratio": float(top_10_ratio),
    }


def crossing_score_from_metrics(
    metrics: FloatDict,
) -> float:
    """
    Convert normalized crossing count into a score in [0, 1].

    Higher is better.
    """
    normalized_crossings = float(metrics.get("normalized_crossing_count", 0.0))
    score = 1.0 - normalized_crossings

    return float(max(0.0, min(1.0, score)))