from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from actions.base_action_space import ActionResult
from envs.layout_context import LayoutContext
from features.normalizers import safe_float
from rewards.base_reward import BaseRewardFunction, FloatDict, RewardResult


EPSILON = 1e-9


@dataclass(frozen=True)
class AestheticRewardWeights:
    """
    Weights for balanced aesthetic improvement.

    These weights are applied to score deltas:

        crossing_score
        angular_resolution_score
        edge_length_score
        node_separation_score
    """

    crossing: float = 0.35
    angular_resolution: float = 0.25
    edge_length: float = 0.20
    node_separation: float = 0.20

    def normalized(self) -> "AestheticRewardWeights":
        total = (
            self.crossing
            + self.angular_resolution
            + self.edge_length
            + self.node_separation
        )

        if total <= 0.0:
            return AestheticRewardWeights(
                crossing=0.25,
                angular_resolution=0.25,
                edge_length=0.25,
                node_separation=0.25,
            )

        return AestheticRewardWeights(
            crossing=self.crossing / total,
            angular_resolution=self.angular_resolution / total,
            edge_length=self.edge_length / total,
            node_separation=self.node_separation / total,
        )


@dataclass(frozen=True)
class AestheticDeltaRewardConfig:
    """
    Configuration for the redesigned reward.

    expansion_penalty_weight:
        Penalizes layouts that expand too much relative to layout_scale.

    repeated_action_penalty_weight:
        Penalizes repeating the same action too many times.

    action_change_penalty_weight:
        Small penalty for changing parameters, encouraging simpler policies.

    terminal_bonus_weight:
        Bonus based on final layout quality at terminal step.

    max_layout_diagonal_factor:
        Allowed layout diagonal relative to layout_scale before expansion penalty.

    repeated_action_threshold:
        Number of repeated actions allowed before penalty begins.

    negative_delta_multiplier:
        If > 1, worsened aesthetics are punished more strongly.
    """

    weights: AestheticRewardWeights = AestheticRewardWeights()

    expansion_penalty_weight: float = 0.05
    repeated_action_penalty_weight: float = 0.02
    action_change_penalty_weight: float = 0.001
    terminal_bonus_weight: float = 0.10

    max_layout_diagonal_factor: float = 8.0
    repeated_action_threshold: int = 3
    negative_delta_multiplier: float = 1.0


class AestheticDeltaReward(BaseRewardFunction):
    """
    Redesigned reward for SARL-based force-directed parameter control.

    Reward structure:

        r_t =
            balanced aesthetic improvement
            - expansion penalty
            - repeated-action penalty
            - action-change penalty
            + terminal quality bonus

    This reward is algorithm-independent. It uses the LayoutContext, not
    Pure-FR-specific logic.
    """

    reward_name = "aesthetic_delta"

    def __init__(
        self,
        config: AestheticDeltaRewardConfig | None = None,
    ):
        self.config = config if config is not None else AestheticDeltaRewardConfig()
        self.weights = self.config.weights.normalized()

    def compute(
        self,
        context: LayoutContext,
        action_result: ActionResult,
        is_terminal: bool,
    ) -> RewardResult:
        crossing_delta = context.get_score_delta("crossing_score", 0.0)
        angular_delta = context.get_score_delta(
            "angular_resolution_score",
            0.0,
        )
        edge_length_delta = context.get_score_delta("edge_length_score", 0.0)
        node_separation_delta = context.get_score_delta(
            "node_separation_score",
            0.0,
        )

        weighted_crossing = self.weights.crossing * self._shape_delta(
            crossing_delta
        )
        weighted_angular = self.weights.angular_resolution * self._shape_delta(
            angular_delta
        )
        weighted_edge_length = self.weights.edge_length * self._shape_delta(
            edge_length_delta
        )
        weighted_node_separation = self.weights.node_separation * self._shape_delta(
            node_separation_delta
        )

        aesthetic_improvement = (
            weighted_crossing
            + weighted_angular
            + weighted_edge_length
            + weighted_node_separation
        )

        expansion_penalty = self._compute_expansion_penalty(context)
        repeated_action_penalty = self._compute_repeated_action_penalty(context)
        action_change_penalty = self._compute_action_change_penalty(action_result)
        terminal_bonus = self._compute_terminal_bonus(
            context=context,
            is_terminal=is_terminal,
        )

        reward = (
            aesthetic_improvement
            - self.config.expansion_penalty_weight * expansion_penalty
            - self.config.repeated_action_penalty_weight * repeated_action_penalty
            - self.config.action_change_penalty_weight * action_change_penalty
            + terminal_bonus
        )

        components: FloatDict = {
            "crossing_delta": float(crossing_delta),
            "angular_resolution_delta": float(angular_delta),
            "edge_length_delta": float(edge_length_delta),
            "node_separation_delta": float(node_separation_delta),
            "weighted_crossing": float(weighted_crossing),
            "weighted_angular_resolution": float(weighted_angular),
            "weighted_edge_length": float(weighted_edge_length),
            "weighted_node_separation": float(weighted_node_separation),
            "aesthetic_improvement": float(aesthetic_improvement),
            "expansion_penalty": float(expansion_penalty),
            "weighted_expansion_penalty": float(
                self.config.expansion_penalty_weight * expansion_penalty
            ),
            "repeated_action_penalty": float(repeated_action_penalty),
            "weighted_repeated_action_penalty": float(
                self.config.repeated_action_penalty_weight
                * repeated_action_penalty
            ),
            "action_change_penalty": float(action_change_penalty),
            "weighted_action_change_penalty": float(
                self.config.action_change_penalty_weight * action_change_penalty
            ),
            "terminal_bonus": float(terminal_bonus),
            "layout_score": float(context.scores.get("layout_score", 0.0)),
            "delta_layout_score": float(
                context.get_score_delta("layout_score", 0.0)
            ),
        }

        return RewardResult(
            reward=float(reward),
            components=components,
        )

    def _shape_delta(
        self,
        delta: float,
    ) -> float:
        """
        Optionally punish negative deltas more strongly.
        """
        delta_float = safe_float(delta)

        if delta_float < 0.0:
            return float(delta_float * self.config.negative_delta_multiplier)

        return float(delta_float)

    def _compute_expansion_penalty(
        self,
        context: LayoutContext,
    ) -> float:
        """
        Penalize excessive layout expansion.

        The layout should not grow without bound just because spacing improves.
        """
        layout_diagonal = safe_float(
            context.layout_stats.get(
                "layout_diagonal",
                context.metrics.get("layout_diagonal", 0.0),
            )
        )

        allowed_diagonal = max(
            EPSILON,
            self.config.max_layout_diagonal_factor
            * max(EPSILON, safe_float(context.layout_scale, 1.0)),
        )

        expansion_ratio = layout_diagonal / allowed_diagonal

        return float(max(0.0, expansion_ratio - 1.0))

    def _compute_repeated_action_penalty(
        self,
        context: LayoutContext,
    ) -> float:
        """
        Penalize repeating the same action beyond a threshold.

        This is intentionally mild. Repetition is not always bad, but the
        prototype showed that repeated increase_k can become a collapse mode.
        """
        repeat_count = context.history.same_action_repeat_count()
        threshold = max(1, int(self.config.repeated_action_threshold))

        if repeat_count <= threshold:
            return 0.0

        excess_repeat = repeat_count - threshold
        denominator = max(1.0, float(context.history.max_length))

        return float(min(1.0, excess_repeat / denominator))

    def _compute_action_change_penalty(
        self,
        action_result: ActionResult,
    ) -> float:
        """
        Small penalty if the action changes parameters.
        """
        return 1.0 if action_result.changed else 0.0

    def _compute_terminal_bonus(
        self,
        context: LayoutContext,
        is_terminal: bool,
    ) -> float:
        """
        Add terminal bonus based on final layout score.
        """
        if not is_terminal:
            return 0.0

        layout_score = safe_float(context.scores.get("layout_score", 0.0))

        return float(self.config.terminal_bonus_weight * layout_score)


class ScoreOnlyReward(BaseRewardFunction):
    """
    Simple baseline reward:

        reward = layout_score_t - layout_score_{t-1}

    This is useful for ablation studies.
    """

    reward_name = "score_only"

    def compute(
        self,
        context: LayoutContext,
        action_result: ActionResult,
        is_terminal: bool,
    ) -> RewardResult:
        _ = action_result
        _ = is_terminal

        delta = context.get_score_delta("layout_score", 0.0)

        components: FloatDict = {
            "delta_layout_score": float(delta),
        }

        return RewardResult(
            reward=float(delta),
            components=components,
        )