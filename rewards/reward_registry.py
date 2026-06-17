from __future__ import annotations

from rewards.aesthetic_delta_reward import (
    AestheticDeltaReward,
    AestheticDeltaRewardConfig,
    AestheticRewardWeights,
    ScoreOnlyReward,
)
from rewards.base_reward import BaseRewardFunction


def create_reward_function(
    reward_name: str,
) -> BaseRewardFunction:
    """
    Create reward function by name.

    This registry supports future ablation studies.
    """
    normalized_name = reward_name.lower().strip()

    if normalized_name in {
        "aesthetic_delta",
        "balanced_aesthetic_delta",
        "default",
    }:
        return AestheticDeltaReward()

    if normalized_name == "score_only":
        return ScoreOnlyReward()

    if normalized_name == "no_expansion_penalty":
        return AestheticDeltaReward(
            config=AestheticDeltaRewardConfig(
                weights=AestheticRewardWeights(),
                expansion_penalty_weight=0.0,
                repeated_action_penalty_weight=0.02,
                action_change_penalty_weight=0.001,
                terminal_bonus_weight=0.10,
            )
        )

    if normalized_name == "no_repeat_penalty":
        return AestheticDeltaReward(
            config=AestheticDeltaRewardConfig(
                weights=AestheticRewardWeights(),
                expansion_penalty_weight=0.05,
                repeated_action_penalty_weight=0.0,
                action_change_penalty_weight=0.001,
                terminal_bonus_weight=0.10,
            )
        )

    if normalized_name == "no_action_penalty":
        return AestheticDeltaReward(
            config=AestheticDeltaRewardConfig(
                weights=AestheticRewardWeights(),
                expansion_penalty_weight=0.05,
                repeated_action_penalty_weight=0.02,
                action_change_penalty_weight=0.0,
                terminal_bonus_weight=0.10,
            )
        )

    available = [
        "aesthetic_delta",
        "balanced_aesthetic_delta",
        "default",
        "score_only",
        "no_expansion_penalty",
        "no_repeat_penalty",
        "no_action_penalty",
    ]

    raise ValueError(
        f"Unknown reward function: {reward_name}. "
        f"Available options: {available}"
    )