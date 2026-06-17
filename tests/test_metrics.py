from typing import Dict, Hashable

import networkx as nx
import numpy as np

from metrics import LayoutQualityEvaluator, LayoutScoreCalculator
from metrics.crossings import compute_crossing_metrics


Node = Hashable
PositionDict = Dict[Node, np.ndarray]


def test_manual_crossing() -> None:
    """
    Test a graph drawing with one crossing.

    Edges:
        0--1
        2--3

    Positions create an X shape.
    """
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])

    positions: PositionDict = {
        0: np.array([0.0, 0.0], dtype=float),
        1: np.array([1.0, 1.0], dtype=float),
        2: np.array([0.0, 1.0], dtype=float),
        3: np.array([1.0, 0.0], dtype=float),
    }

    metrics = compute_crossing_metrics(
        graph=graph,
        positions=positions,
    )

    assert metrics["crossing_count"] == 1.0
    assert metrics["local_crossing_number"] == 1.0

    print("Manual crossing test passed.")
    print(metrics)


def test_layout_quality_and_score() -> None:
    """
    Test the complete metric and scoring pipeline.
    """
    graph = nx.cycle_graph(8)

    positions: PositionDict = {}

    for node in graph.nodes():
        angle = 2.0 * np.pi * int(node) / graph.number_of_nodes()

        positions[node] = np.array(
            [
                np.cos(angle),
                np.sin(angle),
            ],
            dtype=float,
        )

    evaluator = LayoutQualityEvaluator()
    score_calculator = LayoutScoreCalculator()

    metrics = evaluator.evaluate(
        graph=graph,
        positions=positions,
    )

    scores = score_calculator.score(
        metrics=metrics,
    )

    assert "crossing_count" in metrics
    assert "angular_resolution_score" in metrics
    assert "edge_length_score" in metrics
    assert "node_separation_score" in metrics
    assert "layout_score" in scores

    assert 0.0 <= scores["layout_score"] <= 1.0

    print("\nLayout quality test passed.")
    print("Metrics:")

    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    print("Scores:")

    for key, value in scores.items():
        print(f"  {key}: {value:.6f}")


def main() -> None:
    test_manual_crossing()
    test_layout_quality_and_score()


if __name__ == "__main__":
    main()