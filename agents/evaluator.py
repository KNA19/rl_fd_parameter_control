from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO

from envs import FDParamControlEnvConfig, ForceDirectedParameterControlEnv, create_env


ActionSelector = Callable[
    [np.ndarray, ForceDirectedParameterControlEnv, Discrete],
    int,
]


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for model and baseline evaluation.
    """

    metadata_path: str = "data/metadata/dataset_metadata.csv"
    split: str = "val"

    layout_scale: float = 1.0
    max_macro_steps: int = 5
    iterations_per_step: int = 20

    seed: int = 2026

    algorithm_name: str = "fruchterman_reingold"
    state_name: str = "full"
    action_space_name: str = "pure_fr_multiscale"
    reward_name: str = "aesthetic_delta"

    model_path: str = "outputs/models/ppo/fd_param_control_ppo.zip"
    num_episodes: int = 10
    deterministic: bool = True

    output_csv_path: Optional[str] = "outputs/evaluation/evaluation_summary.csv"


def make_evaluation_env(
    config: EvaluationConfig,
) -> ForceDirectedParameterControlEnv:
    env_config = FDParamControlEnvConfig(
        metadata_path=config.metadata_path,
        split=config.split,
        layout_scale=config.layout_scale,
        max_macro_steps=config.max_macro_steps,
        iterations_per_step=config.iterations_per_step,
        seed=config.seed,
        state_name=config.state_name,
        action_space_name=config.action_space_name,
        reward_name=config.reward_name,
        enable_early_stopping=False,
    )

    return create_env(
        config=env_config,
        algorithm_name=config.algorithm_name,
    )


def load_ppo_model(
    model_path: str,
    env: ForceDirectedParameterControlEnv,
) -> PPO:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}")

    return PPO.load(
        path=str(path),
        env=env,
    )


def evaluate_ppo(
    config: EvaluationConfig,
) -> Dict[str, Any]:
    """
    Evaluate a trained PPO model.
    """
    env = make_evaluation_env(config)

    if not isinstance(env.action_space, Discrete):
        raise TypeError("Expected Discrete action space.")

    discrete_action_space = cast(Discrete, env.action_space)

    model = load_ppo_model(
        model_path=config.model_path,
        env=env,
    )

    def ppo_selector(
        observation: np.ndarray,
        current_env: ForceDirectedParameterControlEnv,
        action_space: Discrete,
    ) -> int:
        _ = current_env

        action, _state = model.predict(
            observation,
            deterministic=config.deterministic,
        )

        action_id = int(np.asarray(action).reshape(-1)[0])

        if action_id < 0 or action_id >= int(action_space.n):
            raise ValueError(f"Invalid PPO action: {action_id}")

        return action_id

    summary = _evaluate_policy_with_selector(
        config=config,
        env=env,
        discrete_action_space=discrete_action_space,
        policy_name="ppo",
        action_selector=ppo_selector,
    )

    env.close()

    return summary


def evaluate_random_policy(
    config: EvaluationConfig,
) -> Dict[str, Any]:
    """
    Evaluate a random action policy.
    """
    env = make_evaluation_env(config)

    if not isinstance(env.action_space, Discrete):
        raise TypeError("Expected Discrete action space.")

    discrete_action_space = cast(Discrete, env.action_space)

    def random_selector(
        observation: np.ndarray,
        current_env: ForceDirectedParameterControlEnv,
        action_space: Discrete,
    ) -> int:
        _ = observation
        _ = current_env
        return int(action_space.sample())

    summary = _evaluate_policy_with_selector(
        config=config,
        env=env,
        discrete_action_space=discrete_action_space,
        policy_name="random",
        action_selector=random_selector,
    )

    env.close()

    return summary


def evaluate_fixed_action_policy(
    config: EvaluationConfig,
    action_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a fixed-action policy.

    Examples:
        action_name = "no_change"
        action_name = "large_decrease_k"
    """
    env = make_evaluation_env(config)

    if not isinstance(env.action_space, Discrete):
        raise TypeError("Expected Discrete action space.")

    discrete_action_space = cast(Discrete, env.action_space)

    fixed_action_id_cache: Dict[str, int] = {}

    def fixed_action_selector(
        observation: np.ndarray,
        current_env: ForceDirectedParameterControlEnv,
        action_space: Discrete,
    ) -> int:
        _ = observation
        _ = action_space

        if "action_id" not in fixed_action_id_cache:
            fixed_action_id_cache["action_id"] = get_action_id_by_name(
                env=current_env,
                action_name=action_name,
            )

        return fixed_action_id_cache["action_id"]

    policy_name = f"fixed::{action_name}"

    summary = _evaluate_policy_with_selector(
        config=config,
        env=env,
        discrete_action_space=discrete_action_space,
        policy_name=policy_name,
        action_selector=fixed_action_selector,
    )

    env.close()

    return summary


def get_action_id_by_name(
    env: ForceDirectedParameterControlEnv,
    action_name: str,
) -> int:
    """
    Return action id for a given action name.

    The environment must already be reset because parameter_action_space is
    created during reset().
    """
    if env.parameter_action_space is None:
        raise RuntimeError(
            "parameter_action_space is not initialized. "
            "Call env.reset() before getting an action id."
        )

    action_names = list(env.parameter_action_space.action_names)

    if action_name not in action_names:
        available = ", ".join(action_names)
        raise ValueError(
            f"Action name '{action_name}' not found. "
            f"Available actions: {available}"
        )

    return int(action_names.index(action_name))


def _evaluate_policy_with_selector(
    config: EvaluationConfig,
    env: ForceDirectedParameterControlEnv,
    discrete_action_space: Discrete,
    policy_name: str,
    action_selector: ActionSelector,
) -> Dict[str, Any]:
    """
    Shared evaluation loop for PPO, random, no-change, and fixed-action policies.
    """
    episode_rows: List[Dict[str, Any]] = []
    action_counter: Counter[str] = Counter()

    total_rewards: List[float] = []
    initial_scores: List[float] = []
    final_scores: List[float] = []
    improvements: List[float] = []
    episode_lengths: List[int] = []

    num_available_graphs = len(env.metadata_rows)

    for episode_index in range(config.num_episodes):
        graph_index = episode_index % num_available_graphs

        observation, info = env.reset(
            seed=config.seed + episode_index,
            options={"graph_index": graph_index},
        )

        initial_score = float(info.get("layout_score", 0.0))
        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        final_info = info

        while not terminated and not truncated:
            action_id = action_selector(
                observation,
                env,
                discrete_action_space,
            )

            if action_id < 0 or action_id >= int(discrete_action_space.n):
                raise ValueError(f"Invalid action id: {action_id}")

            observation, reward, terminated, truncated, step_info = env.step(
                action_id
            )

            total_reward += float(reward)
            step_count += 1
            final_info = step_info

            action_name_selected = str(step_info.get("action_name", "unknown"))
            action_counter[action_name_selected] += 1

        final_score = float(final_info.get("layout_score", 0.0))
        improvement = final_score - initial_score

        row = {
            "policy": policy_name,
            "episode": episode_index,
            "graph_id": final_info.get("graph_id", "unknown"),
            "family": final_info.get("family", "unknown"),
            "size_label": final_info.get("size_label", "unknown"),
            "initial_layout_score": initial_score,
            "final_layout_score": final_score,
            "layout_score_improvement": improvement,
            "total_reward": total_reward,
            "episode_length": step_count,
        }

        episode_rows.append(row)

        total_rewards.append(total_reward)
        initial_scores.append(initial_score)
        final_scores.append(final_score)
        improvements.append(improvement)
        episode_lengths.append(step_count)

    summary: Dict[str, Any] = {
        "policy": policy_name,
        "split": config.split,
        "num_episodes": config.num_episodes,
        "mean_total_reward": float(np.mean(total_rewards)),
        "mean_initial_layout_score": float(np.mean(initial_scores)),
        "mean_final_layout_score": float(np.mean(final_scores)),
        "mean_layout_score_improvement": float(np.mean(improvements)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "action_counts": dict(action_counter),
        "episode_rows": episode_rows,
    }

    if config.output_csv_path is not None:
        save_evaluation_csv(
            rows=episode_rows,
            output_csv_path=config.output_csv_path,
        )

    print_evaluation_summary(summary)

    return summary


def save_evaluation_csv(
    rows: List[Dict[str, Any]],
    output_csv_path: str,
) -> None:
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation CSV to: {output_path}")


def save_comparison_csv(
    summaries: List[Dict[str, Any]],
    output_csv_path: str,
) -> None:
    """
    Save one-row-per-policy comparison CSV.
    """
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for summary in summaries:
        rows.append(
            {
                "policy": summary["policy"],
                "split": summary["split"],
                "num_episodes": summary["num_episodes"],
                "mean_total_reward": summary["mean_total_reward"],
                "mean_initial_layout_score": summary[
                    "mean_initial_layout_score"
                ],
                "mean_final_layout_score": summary["mean_final_layout_score"],
                "mean_layout_score_improvement": summary[
                    "mean_layout_score_improvement"
                ],
                "mean_episode_length": summary["mean_episode_length"],
                "action_counts": str(summary.get("action_counts", {})),
            }
        )

    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison CSV to: {output_path}")


def print_evaluation_summary(
    summary: Dict[str, Any],
) -> None:
    print("\nEvaluation summary")
    print("------------------")
    print(f"Policy: {summary['policy']}")
    print(f"Split: {summary['split']}")
    print(f"Episodes: {summary['num_episodes']}")
    print(f"Mean total reward: {summary['mean_total_reward']:.6f}")
    print(f"Mean initial layout score: {summary['mean_initial_layout_score']:.6f}")
    print(f"Mean final layout score: {summary['mean_final_layout_score']:.6f}")
    print(
        "Mean layout-score improvement: "
        f"{summary['mean_layout_score_improvement']:.6f}"
    )
    print(f"Mean episode length: {summary['mean_episode_length']:.2f}")
    print("Action counts:")

    action_counts = summary.get("action_counts", {})

    for action_name, count in action_counts.items():
        print(f"  {action_name}: {count}")


def print_policy_comparison(
    summaries: List[Dict[str, Any]],
) -> None:
    """
    Print compact comparison table.
    """
    print("\nPolicy comparison")
    print("-----------------")
    print(
        f"{'Policy':30s} | "
        f"{'Initial':>10s} | "
        f"{'Final':>10s} | "
        f"{'Improve':>10s} | "
        f"{'Reward':>10s}"
    )
    print("-" * 82)

    for summary in summaries:
        print(
            f"{summary['policy']:30s} | "
            f"{summary['mean_initial_layout_score']:10.6f} | "
            f"{summary['mean_final_layout_score']:10.6f} | "
            f"{summary['mean_layout_score_improvement']:10.6f} | "
            f"{summary['mean_total_reward']:10.6f}"
        )