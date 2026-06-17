from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO

from agents.evaluator import get_action_id_by_name
from envs import FDParamControlEnvConfig, ForceDirectedParameterControlEnv, create_env


PositionDict = Dict[Any, np.ndarray]
FloatDict = Dict[str, float]


@dataclass
class PolicyLayoutRunResult:
    """
    Stores one complete layout run under one policy.
    """

    policy_name: str
    graph_id: str
    family: str
    size_label: str

    graph: nx.Graph

    initial_positions: PositionDict
    final_positions: PositionDict

    initial_score: float
    final_score: float
    improvement: float
    total_reward: float

    initial_metrics: FloatDict = field(default_factory=dict)
    final_metrics: FloatDict = field(default_factory=dict)

    action_sequence: List[str] = field(default_factory=list)
    reward_sequence: List[float] = field(default_factory=list)
    score_sequence: List[float] = field(default_factory=list)

    final_info: Dict[str, Any] = field(default_factory=dict)


def run_policy_on_graph(
    config: FDParamControlEnvConfig,
    graph_index: int,
    policy_name: str,
    seed: int,
    model_path: Optional[str] = None,
    fixed_action_name: Optional[str] = None,
    algorithm_name: str = "fruchterman_reingold",
) -> PolicyLayoutRunResult:
    """
    Run one policy on one graph.

    Supported policy_name values:
        no_change
        random
        fixed
        ppo

    For fixed policy:
        fixed_action_name must be provided, e.g. "large_decrease_k".

    For PPO policy:
        model_path must point to a saved Stable-Baselines3 PPO model.
    """
    env = create_env(
        config=config,
        algorithm_name=algorithm_name,
    )

    if not isinstance(env.action_space, Discrete):
        raise TypeError("Expected env.action_space to be gymnasium.spaces.Discrete.")

    discrete_action_space = cast(Discrete, env.action_space)
    discrete_action_space.seed(seed + 999)

    observation, info = env.reset(
        seed=seed,
        options={"graph_index": graph_index},
    )

    if env.context is None:
        raise RuntimeError("Environment context was not initialized after reset().")

    if env.current_graph is None:
        raise RuntimeError("Environment graph was not initialized after reset().")

    graph = nx.Graph(env.current_graph)

    initial_positions = _copy_positions(env.context.positions)
    initial_score = float(info.get("layout_score", 0.0))
    initial_metrics = dict(env.context.metrics)

    model: Optional[PPO] = None

    normalized_policy_name = policy_name.lower().strip()

    if normalized_policy_name == "ppo":
        if model_path is None:
            raise ValueError("model_path is required for PPO policy.")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"PPO model not found: {model_path}")

        model = PPO.load(
            path=model_path,
            env=env,
        )

    if normalized_policy_name == "fixed":
        if fixed_action_name is None:
            raise ValueError("fixed_action_name is required for fixed policy.")

        selected_policy_label = f"fixed::{fixed_action_name}"
    else:
        selected_policy_label = normalized_policy_name

    terminated = False
    truncated = False
    total_reward = 0.0

    action_sequence: List[str] = []
    reward_sequence: List[float] = []
    score_sequence: List[float] = [initial_score]

    final_info: Dict[str, Any] = dict(info)

    while not terminated and not truncated:
        action_id = _select_action(
            policy_name=normalized_policy_name,
            observation=observation,
            env=env,
            discrete_action_space=discrete_action_space,
            model=model,
            fixed_action_name=fixed_action_name,
        )

        observation, reward, terminated, truncated, step_info = env.step(action_id)

        action_name = str(step_info.get("action_name", "unknown"))
        layout_score = float(step_info.get("layout_score", 0.0))

        action_sequence.append(action_name)
        reward_sequence.append(float(reward))
        score_sequence.append(layout_score)

        total_reward += float(reward)
        final_info = dict(step_info)

    if env.context is None:
        raise RuntimeError("Environment context became unavailable.")

    final_positions = _copy_positions(env.context.positions)
    final_score = float(env.context.scores.get("layout_score", 0.0))
    final_metrics = dict(env.context.metrics)

    result = PolicyLayoutRunResult(
        policy_name=selected_policy_label,
        graph_id=str(final_info.get("graph_id", env.context.graph_id)),
        family=str(final_info.get("family", "unknown")),
        size_label=str(final_info.get("size_label", "unknown")),
        graph=graph,
        initial_positions=initial_positions,
        final_positions=final_positions,
        initial_score=initial_score,
        final_score=final_score,
        improvement=final_score - initial_score,
        total_reward=total_reward,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        action_sequence=action_sequence,
        reward_sequence=reward_sequence,
        score_sequence=score_sequence,
        final_info=final_info,
    )

    env.close()

    return result


def _select_action(
    policy_name: str,
    observation: np.ndarray,
    env: ForceDirectedParameterControlEnv,
    discrete_action_space: Discrete,
    model: Optional[PPO],
    fixed_action_name: Optional[str],
) -> int:
    """
    Select action for one environment step.
    """
    if policy_name == "no_change":
        return get_action_id_by_name(
            env=env,
            action_name="no_change",
        )

    if policy_name == "random":
        return int(discrete_action_space.sample())

    if policy_name == "fixed":
        if fixed_action_name is None:
            raise ValueError("fixed_action_name is required for fixed policy.")

        return get_action_id_by_name(
            env=env,
            action_name=fixed_action_name,
        )

    if policy_name == "ppo":
        if model is None:
            raise ValueError("model is required for PPO policy.")

        action, _state = model.predict(
            observation,
            deterministic=True,
        )

        action_id = int(np.asarray(action).reshape(-1)[0])

        if action_id < 0 or action_id >= int(discrete_action_space.n):
            raise ValueError(f"Invalid PPO action id: {action_id}")

        return action_id

    raise ValueError(
        f"Unknown policy_name={policy_name}. "
        "Available policies: no_change, random, fixed, ppo."
    )


def _copy_positions(
    positions: Mapping[Any, np.ndarray],
) -> PositionDict:
    copied: PositionDict = {}

    for node, position in positions.items():
        copied[node] = np.asarray(position, dtype=float).copy()

    return copied