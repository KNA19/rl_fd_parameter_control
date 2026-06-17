from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


FloatDict = Dict[str, float]


@dataclass
class HistoryEntry:
    """
    One recorded environment step.

    This is algorithm-independent. It can store Pure FR history now,
    and later Eades, Kamada-Kawai, or other force-directed algorithms.
    """

    step_index: int
    action_id: int
    action_name: str
    parameters: FloatDict = field(default_factory=dict)
    metrics: FloatDict = field(default_factory=dict)
    scores: FloatDict = field(default_factory=dict)
    layout_stats: FloatDict = field(default_factory=dict)

    def get_metric(self, name: str, default: float = 0.0) -> float:
        return float(self.metrics.get(name, default))

    def get_score(self, name: str, default: float = 0.0) -> float:
        return float(self.scores.get(name, default))

    def get_parameter(self, name: str, default: float = 0.0) -> float:
        return float(self.parameters.get(name, default))

    def get_layout_stat(self, name: str, default: float = 0.0) -> float:
        return float(self.layout_stats.get(name, default))


class HistoryBuffer:
    """
    Fixed-length buffer that stores recent actions, metrics, scores,
    parameters, and layout statistics.

    The redesigned state will later use this for:

        last action
        repeated action count
        recent action counts
        metric deltas
        score deltas
        parameter deltas
        previous displacement information
    """

    def __init__(
        self,
        max_length: int = 20,
        action_names: Optional[List[str]] = None,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.max_length = int(max_length)
        self.action_names = action_names if action_names is not None else []
        self.entries: Deque[HistoryEntry] = deque(maxlen=self.max_length)

    def reset(self) -> None:
        """
        Clear all stored history.
        """
        self.entries.clear()

    def append(
        self,
        step_index: int,
        action_id: int,
        action_name: str,
        parameters: Optional[FloatDict] = None,
        metrics: Optional[FloatDict] = None,
        scores: Optional[FloatDict] = None,
        layout_stats: Optional[FloatDict] = None,
    ) -> None:
        """
        Add one history entry.
        """
        entry = HistoryEntry(
            step_index=int(step_index),
            action_id=int(action_id),
            action_name=str(action_name),
            parameters=self._copy_float_dict(parameters),
            metrics=self._copy_float_dict(metrics),
            scores=self._copy_float_dict(scores),
            layout_stats=self._copy_float_dict(layout_stats),
        )

        self.entries.append(entry)

    def last(self) -> Optional[HistoryEntry]:
        """
        Return the most recent entry.
        """
        if not self.entries:
            return None

        return self.entries[-1]

    def previous(self) -> Optional[HistoryEntry]:
        """
        Return the second most recent entry.
        """
        if len(self.entries) < 2:
            return None

        return self.entries[-2]

    def size(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def get_recent_entries(self, window: int) -> List[HistoryEntry]:
        """
        Return the most recent entries up to a fixed window size.
        """
        if window <= 0:
            return []

        return list(self.entries)[-window:]

    def last_action_id(self, default: int = -1) -> int:
        entry = self.last()

        if entry is None:
            return default

        return int(entry.action_id)

    def last_action_name(self, default: str = "none") -> str:
        entry = self.last()

        if entry is None:
            return default

        return str(entry.action_name)

    def same_action_repeat_count(self) -> int:
        """
        Count how many times the latest action has been repeated consecutively.

        Example:
            actions = increase_k, decrease_k, increase_k, increase_k
            repeat count = 2
        """
        last_entry = self.last()

        if last_entry is None:
            return 0

        target_action = last_entry.action_name
        count = 0

        for entry in reversed(self.entries):
            if entry.action_name == target_action:
                count += 1
            else:
                break

        return count

    def same_action_repeat_count_norm(self) -> float:
        """
        Normalize repeated action count to [0, 1].
        """
        if self.max_length <= 0:
            return 0.0

        value = self.same_action_repeat_count() / self.max_length
        return float(max(0.0, min(1.0, value)))

    def last_action_one_hot(self, num_actions: int) -> np.ndarray:
        """
        Return one-hot encoding of the last action.

        If there is no last action, return all zeros.
        """
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")

        vector = np.zeros(num_actions, dtype=np.float32)
        last_action = self.last_action_id(default=-1)

        if 0 <= last_action < num_actions:
            vector[last_action] = 1.0

        return vector

    def recent_action_count(
        self,
        action_name: str,
        window: Optional[int] = None,
    ) -> int:
        """
        Count how many times an action appears in the recent window.
        """
        if window is None:
            entries = list(self.entries)
        else:
            entries = self.get_recent_entries(window)

        return sum(1 for entry in entries if entry.action_name == action_name)

    def recent_action_fraction(
        self,
        action_name: str,
        window: Optional[int] = None,
    ) -> float:
        """
        Fraction of recent actions that match action_name.
        """
        if window is None:
            entries = list(self.entries)
        else:
            entries = self.get_recent_entries(window)

        if not entries:
            return 0.0

        count = sum(1 for entry in entries if entry.action_name == action_name)

        return float(count / len(entries))

    def metric_delta(
        self,
        metric_name: str,
        default: float = 0.0,
    ) -> float:
        """
        Return latest metric value minus previous metric value.
        """
        latest = self.last()
        previous = self.previous()

        if latest is None or previous is None:
            return default

        return latest.get_metric(metric_name) - previous.get_metric(metric_name)

    def score_delta(
        self,
        score_name: str,
        default: float = 0.0,
    ) -> float:
        """
        Return latest score value minus previous score value.
        """
        latest = self.last()
        previous = self.previous()

        if latest is None or previous is None:
            return default

        return latest.get_score(score_name) - previous.get_score(score_name)

    def parameter_delta(
        self,
        parameter_name: str,
        default: float = 0.0,
    ) -> float:
        """
        Return latest parameter value minus previous parameter value.
        """
        latest = self.last()
        previous = self.previous()

        if latest is None or previous is None:
            return default

        return latest.get_parameter(parameter_name) - previous.get_parameter(
            parameter_name
        )

    def layout_stat_delta(
        self,
        stat_name: str,
        default: float = 0.0,
    ) -> float:
        """
        Return latest layout statistic minus previous layout statistic.
        """
        latest = self.last()
        previous = self.previous()

        if latest is None or previous is None:
            return default

        return latest.get_layout_stat(stat_name) - previous.get_layout_stat(
            stat_name
        )

    def action_summary(
        self,
        window: Optional[int] = None,
    ) -> FloatDict:
        """
        Return normalized recent action frequencies.

        Keys are:
            action_fraction::<action_name>
        """
        summary: FloatDict = {}

        if not self.action_names:
            return summary

        if window is None:
            entries = list(self.entries)
        else:
            entries = self.get_recent_entries(window)

        denominator = max(1, len(entries))

        for action_name in self.action_names:
            count = sum(1 for entry in entries if entry.action_name == action_name)
            summary[f"action_fraction::{action_name}"] = float(
                count / denominator
            )

        return summary

    def to_debug_dict(self) -> Dict[str, float | int | str]:
        """
        Return compact debug information.
        """
        return {
            "history_size": self.size(),
            "last_action_id": self.last_action_id(),
            "last_action_name": self.last_action_name(),
            "same_action_repeat_count": self.same_action_repeat_count(),
            "same_action_repeat_count_norm": self.same_action_repeat_count_norm(),
        }

    def _copy_float_dict(
        self,
        data: Optional[FloatDict],
    ) -> FloatDict:
        if data is None:
            return {}

        copied: FloatDict = {}

        for key, value in data.items():
            try:
                copied[str(key)] = float(value)
            except (TypeError, ValueError):
                copied[str(key)] = 0.0

        return copied