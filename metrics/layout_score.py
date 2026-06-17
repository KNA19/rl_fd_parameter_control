from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


FloatDict = Dict[str, float]


@dataclass(frozen=True)
class LayoutScoreWeights:
    """
    Weights for combining layout-quality sub-scores.

    These can later be moved into a YAML config file.
    """

    crossing: float = 0.35
    angular_resolution: float = 0.25
    edge_length: float = 0.20
    node_separation: float = 0.20

    def normalized(self) -> "LayoutScoreWeights":
        total = (
            self.crossing
            + self.angular_resolution
            + self.edge_length
            + self.node_separation
        )

        if total <= 0.0:
            return LayoutScoreWeights(
                crossing=0.25,
                angular_resolution=0.25,
                edge_length=0.25,
                node_separation=0.25,
            )

        return LayoutScoreWeights(
            crossing=self.crossing / total,
            angular_resolution=self.angular_resolution / total,
            edge_length=self.edge_length / total,
            node_separation=self.node_separation / total,
        )


class LayoutScoreCalculator:
    """
    Computes the final normalized layout score.

    The score is in [0, 1], where higher is better.
    """

    def __init__(
        self,
        weights: LayoutScoreWeights | None = None,
    ):
        self.weights = (
            weights if weights is not None else LayoutScoreWeights()
        ).normalized()

    def score(
        self,
        metrics: Mapping[str, float],
    ) -> FloatDict:
        crossing_score = self._clip01(
            metrics.get("crossing_score", 0.0)
        )

        angular_score = self._clip01(
            metrics.get("angular_resolution_score", 0.0)
        )

        edge_length_score = self._clip01(
            metrics.get("edge_length_score", 0.0)
        )

        node_separation_score = self._clip01(
            metrics.get("node_separation_score", 0.0)
        )

        layout_score = (
            self.weights.crossing * crossing_score
            + self.weights.angular_resolution * angular_score
            + self.weights.edge_length * edge_length_score
            + self.weights.node_separation * node_separation_score
        )

        layout_score = self._clip01(layout_score)

        return {
            "layout_score": float(layout_score),
            "crossing_score": float(crossing_score),
            "angular_resolution_score": float(angular_score),
            "edge_length_score": float(edge_length_score),
            "node_separation_score": float(node_separation_score),
        }

    def score_with_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> FloatDict:
        """
        Return a combined dictionary containing both raw metrics and scores.
        """
        output: FloatDict = {
            str(key): float(value) for key, value in metrics.items()
        }

        output.update(self.score(metrics))

        return output

    @staticmethod
    def _clip01(value: float) -> float:
        value_float = float(value)

        if value_float < 0.0:
            return 0.0

        if value_float > 1.0:
            return 1.0

        return value_float