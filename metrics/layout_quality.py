from __future__ import annotations

from typing import Dict, Hashable, Mapping

import networkx as nx
import numpy as np

from metrics.angular_resolution import (
    angular_resolution_score_from_metrics,
    compute_angular_resolution_metrics,
)
from metrics.crossings import (
    compute_crossing_metrics,
    crossing_score_from_metrics,
)
from metrics.edge_lengths import (
    compute_edge_length_metrics,
    edge_length_score_from_metrics,
)
from metrics.node_separation import (
    compute_node_separation_metrics,
    node_separation_score_from_metrics,
)


Node = Hashable
PositionDict = Mapping[Node, np.ndarray]
FloatDict = Dict[str, float]


class LayoutQualityEvaluator:
    """
    General layout-quality evaluator.

    It computes raw metrics and normalized sub-scores.

    Metrics included:

        crossing metrics
        angular-resolution metrics
        edge-length metrics
        node-separation metrics

    The final layout score is computed separately in layout_score.py.
    """

    def evaluate(
        self,
        graph: nx.Graph,
        positions: PositionDict,
    ) -> FloatDict:
        metrics: FloatDict = {}

        crossing_metrics = compute_crossing_metrics(
            graph=graph,
            positions=positions,
        )

        angular_metrics = compute_angular_resolution_metrics(
            graph=graph,
            positions=positions,
        )

        edge_length_metrics = compute_edge_length_metrics(
            graph=graph,
            positions=positions,
        )

        node_separation_metrics = compute_node_separation_metrics(
            graph=graph,
            positions=positions,
        )

        metrics.update(crossing_metrics)
        metrics.update(angular_metrics)
        metrics.update(edge_length_metrics)
        metrics.update(node_separation_metrics)

        metrics["crossing_score"] = crossing_score_from_metrics(metrics)
        metrics["angular_resolution_score"] = (
            angular_resolution_score_from_metrics(metrics)
        )
        metrics["edge_length_score"] = edge_length_score_from_metrics(metrics)
        metrics["node_separation_score"] = (
            node_separation_score_from_metrics(metrics)
        )

        return metrics