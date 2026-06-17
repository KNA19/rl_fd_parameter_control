from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

from actions.base_action_space import ActionResult
from envs.layout_context import LayoutContext


FloatDict = Dict[str, float]


@dataclass(frozen=True)
class RewardResult:
    """
    Reward output returned by a reward function.

    reward:
        Final scalar reward given to the RL agent.

    components:
        Breakdown of reward terms for debugging, analysis, and paper reporting.
    """

    reward: float
    components: FloatDict = field(default_factory=dict)

    def to_info_dict(self) -> FloatDict:
        output: FloatDict = {
            "reward": float(self.reward),
        }

        for key, value in self.components.items():
            output[f"reward_component::{key}"] = float(value)

        return output


class BaseRewardFunction(ABC):
    """
    Base interface for all reward functions.

    Any reward used in the framework should follow this interface.
    """

    reward_name: str = "base_reward"

    @abstractmethod
    def compute(
        self,
        context: LayoutContext,
        action_result: ActionResult,
        is_terminal: bool,
    ) -> RewardResult:
        """
        Compute reward from the current LayoutContext and ActionResult.

        Important:
            The context should already be updated to the new layout state.
            Therefore, context.previous_scores and context.scores represent
            old and new layout conditions.
        """
        raise NotImplementedError