from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, Mapping, Optional

import networkx as nx
import numpy as np

from algorithms.base import LayoutStats, ParameterDict, PositionDict, copy_positions
from utils.history_buffer import HistoryBuffer


Node = Hashable
FloatDict = Dict[str, float]


@dataclass
class LayoutContext:
    """
    Stores the current and previous layout situation.

    The RL environment will update this after every parameter-control action.

    The redesigned state builder will later read from this context to build:

        graph features
        parameter features
        layout metric features
        metric deltas
        layout dynamics
        conflict features
        action-history features
    """

    graph: nx.Graph
    algorithm_name: str
    layout_scale: float
    max_steps: int

    positions: PositionDict
    parameters: ParameterDict

    metrics: FloatDict = field(default_factory=dict)
    scores: FloatDict = field(default_factory=dict)
    layout_stats: FloatDict = field(default_factory=dict)

    previous_positions: Optional[PositionDict] = None
    previous_parameters: Optional[ParameterDict] = None
    previous_metrics: Optional[FloatDict] = None
    previous_scores: Optional[FloatDict] = None
    previous_layout_stats: Optional[FloatDict] = None

    current_step: int = 0
    last_action_id: int = -1
    last_action_name: str = "none"

    history: HistoryBuffer = field(default_factory=HistoryBuffer)

    graph_id: str = ""

    @classmethod
    def create_initial(
        cls,
        graph: nx.Graph,
        algorithm_name: str,
        positions: Mapping[Node, np.ndarray],
        parameters: Mapping[str, float],
        layout_scale: float,
        max_steps: int,
        metrics: Optional[FloatDict] = None,
        scores: Optional[FloatDict] = None,
        layout_stats: Optional[FloatDict] = None,
        action_names: Optional[list[str]] = None,
        graph_id: str = "",
    ) -> "LayoutContext":
        """
        Create context at the start of an episode/layout run.
        """
        history = HistoryBuffer(
            max_length=max(20, max_steps + 5),
            action_names=action_names,
        )

        context = cls(
            graph=graph,
            algorithm_name=algorithm_name,
            layout_scale=float(layout_scale),
            max_steps=int(max_steps),
            positions=copy_positions(positions),
            parameters=cls._copy_float_dict(parameters),
            metrics=cls._copy_float_dict(metrics),
            scores=cls._copy_float_dict(scores),
            layout_stats=cls._copy_float_dict(layout_stats),
            current_step=0,
            history=history,
            graph_id=graph_id,
        )

        history.append(
            step_index=0,
            action_id=-1,
            action_name="initial",
            parameters=context.parameters,
            metrics=context.metrics,
            scores=context.scores,
            layout_stats=context.layout_stats,
        )

        return context

    def update_after_step(
        self,
        action_id: int,
        action_name: str,
        new_positions: Mapping[Node, np.ndarray],
        new_parameters: Mapping[str, float],
        new_metrics: Optional[FloatDict] = None,
        new_scores: Optional[FloatDict] = None,
        new_layout_stats: Optional[FloatDict | LayoutStats] = None,
    ) -> None:
        """
        Update the context after one environment macro-step.
        """
        self.previous_positions = copy_positions(self.positions)
        self.previous_parameters = self.parameters.copy()
        self.previous_metrics = self.metrics.copy()
        self.previous_scores = self.scores.copy()
        self.previous_layout_stats = self.layout_stats.copy()

        self.positions = copy_positions(new_positions)
        self.parameters = self._copy_float_dict(new_parameters)
        self.metrics = self._copy_float_dict(new_metrics)
        self.scores = self._copy_float_dict(new_scores)
        self.layout_stats = self._layout_stats_to_dict(new_layout_stats)

        self.current_step += 1
        self.last_action_id = int(action_id)
        self.last_action_name = str(action_name)

        self.history.append(
            step_index=self.current_step,
            action_id=self.last_action_id,
            action_name=self.last_action_name,
            parameters=self.parameters,
            metrics=self.metrics,
            scores=self.scores,
            layout_stats=self.layout_stats,
        )

    @property
    def progress(self) -> float:
        """
        Normalized progress in [0, 1].
        """
        if self.max_steps <= 0:
            return 0.0

        value = self.current_step / self.max_steps
        return float(max(0.0, min(1.0, value)))

    @property
    def is_terminal(self) -> bool:
        """
        Return True if the current episode/layout run has reached max_steps.
        """
        return self.current_step >= self.max_steps

    def get_metric_delta(
        self,
        metric_name: str,
        default: float = 0.0,
    ) -> float:
        """
        Current metric minus previous metric.
        """
        if self.previous_metrics is None:
            return default

        current_value = float(self.metrics.get(metric_name, default))
        previous_value = float(self.previous_metrics.get(metric_name, default))

        return current_value - previous_value

    def get_score_delta(
        self,
        score_name: str,
        default: float = 0.0,
    ) -> float:
        """
        Current score minus previous score.
        """
        if self.previous_scores is None:
            return default

        current_value = float(self.scores.get(score_name, default))
        previous_value = float(self.previous_scores.get(score_name, default))

        return current_value - previous_value

    def get_parameter_delta(
        self,
        parameter_name: str,
        default: float = 0.0,
    ) -> float:
        """
        Current parameter minus previous parameter.
        """
        if self.previous_parameters is None:
            return default

        current_value = float(self.parameters.get(parameter_name, default))
        previous_value = float(
            self.previous_parameters.get(parameter_name, default)
        )

        return current_value - previous_value

    def get_layout_stat_delta(
        self,
        stat_name: str,
        default: float = 0.0,
    ) -> float:
        """
        Current layout statistic minus previous layout statistic.
        """
        if self.previous_layout_stats is None:
            return default

        current_value = float(self.layout_stats.get(stat_name, default))
        previous_value = float(
            self.previous_layout_stats.get(stat_name, default)
        )

        return current_value - previous_value

    def get_position_displacement_summary(self) -> FloatDict:
        """
        Compute displacement summary from previous_positions to positions.

        This is independent of algorithm-reported LayoutStats and can be used
        later by the redesigned state builder.
        """
        if self.previous_positions is None:
            return {
                "context_mean_displacement": 0.0,
                "context_max_displacement": 0.0,
                "context_total_displacement": 0.0,
            }

        displacements = []

        for node, current_position in self.positions.items():
            previous_position = self.previous_positions.get(node)

            if previous_position is None:
                continue

            displacement = float(
                np.linalg.norm(
                    np.asarray(current_position, dtype=float)
                    - np.asarray(previous_position, dtype=float)
                )
            )

            displacements.append(displacement)

        if not displacements:
            return {
                "context_mean_displacement": 0.0,
                "context_max_displacement": 0.0,
                "context_total_displacement": 0.0,
            }

        return {
            "context_mean_displacement": float(np.mean(displacements)),
            "context_max_displacement": float(np.max(displacements)),
            "context_total_displacement": float(np.sum(displacements)),
        }

    def to_summary_dict(self) -> Dict[str, float | int | str]:
        """
        Compact summary for debugging and tests.
        """
        summary: Dict[str, float | int | str] = {
            "algorithm_name": self.algorithm_name,
            "graph_id": self.graph_id,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "progress": self.progress,
            "is_terminal": int(self.is_terminal),
            "last_action_id": self.last_action_id,
            "last_action_name": self.last_action_name,
        }

        if "layout_score" in self.scores:
            summary["layout_score"] = float(self.scores["layout_score"])

        if (
            self.previous_scores is not None
            and "layout_score" in self.previous_scores
        ):
            summary["delta_layout_score"] = self.get_score_delta("layout_score")

        summary.update(self.history.to_debug_dict())

        return summary

    @staticmethod
    def _copy_float_dict(
        data: Optional[Mapping[str, float]],
    ) -> FloatDict:
        """
        Copy a mapping into a float dictionary.
        """
        if data is None:
            return {}

        copied: FloatDict = {}

        for key, value in data.items():
            try:
                copied[str(key)] = float(value)
            except (TypeError, ValueError):
                copied[str(key)] = 0.0

        return copied

    @staticmethod
    def _layout_stats_to_dict(
        stats: Optional[FloatDict | LayoutStats],
    ) -> FloatDict:
        """
        Convert LayoutStats or a float dictionary into a plain float dictionary.
        """
        if stats is None:
            return {}

        if isinstance(stats, LayoutStats):
            return stats.to_dict()

        return LayoutContext._copy_float_dict(stats)