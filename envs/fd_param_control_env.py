from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from actions import BaseParameterActionSpace, create_action_space
from actions.base_action_space import ActionResult
from algorithms.base import BaseForceDirectedAlgorithm, ParameterSpace
from algorithms.fruchterman_reingold import FruchtermanReingoldAlgorithm
from envs.layout_context import LayoutContext
from graph_data.io import MetadataRow, load_graph_pickle, load_metadata_csv
from metrics import LayoutQualityEvaluator, LayoutScoreCalculator
from rewards import BaseRewardFunction, create_reward_function
from states import StateBuilder, create_state_builder


InfoDict = Dict[str, Any]


@dataclass(frozen=True)
class FDParamControlEnvConfig:
    """
    Configuration for the force-directed parameter-control environment.

    This environment uses macro-steps:

        one RL action
        -> update algorithm parameters
        -> run force-directed algorithm for iterations_per_step
        -> compute metrics, reward, and next state

    For final experiments, a useful setting is:

        max_macro_steps = 50
        iterations_per_step = 50

    This gives 2500 total force-directed iterations.
    """

    metadata_path: str = "data/metadata/dataset_metadata.csv"
    split: str = "train"

    layout_scale: float = 1.0
    max_macro_steps: int = 50
    iterations_per_step: int = 50

    seed: int = 2026

    state_name: str = "full"
    action_space_name: str = "pure_fr_multiscale"
    reward_name: str = "aesthetic_delta"

    enable_early_stopping: bool = False
    average_displacement_threshold: float = 0.01
    displacement_rate_threshold: float = 0.005

    graph_id_key: str = "graph_id"
    graph_path_key: str = "graph_path"


class ForceDirectedParameterControlEnv(gym.Env):
    """
    General Gymnasium environment for SARL-based parameter control of
    force-directed algorithms.

    The agent does not move nodes directly. It only chooses parameter-control
    actions. The force-directed algorithm moves the nodes.
    """

    metadata: Dict[str, Any] = {
        "render_modes": [],
    }

    def __init__(
        self,
        config: FDParamControlEnvConfig,
        algorithm: Optional[BaseForceDirectedAlgorithm] = None,
    ):
        super().__init__()

        self.config = config
        self.algorithm = algorithm if algorithm is not None else FruchtermanReingoldAlgorithm()

        self.layout_evaluator = LayoutQualityEvaluator()
        self.score_calculator = LayoutScoreCalculator()
        self.reward_function: BaseRewardFunction = create_reward_function(
            config.reward_name
        )

        self.rng = np.random.default_rng(config.seed)

        self.metadata_rows = self._load_split_metadata(
            metadata_path=config.metadata_path,
            split=config.split,
        )

        self.current_graph: Optional[nx.Graph] = None
        self.current_metadata_row: Optional[MetadataRow] = None

        self.parameter_space: Optional[ParameterSpace] = None
        self.parameter_action_space: Optional[BaseParameterActionSpace] = None
        self.state_builder: Optional[StateBuilder] = None
        self.context: Optional[LayoutContext] = None

        sample_graph, _sample_row = self._sample_graph()
        sample_parameter_space = self.algorithm.get_parameter_space(
            graph=sample_graph,
            layout_scale=self.config.layout_scale,
        )

        sample_action_space = create_action_space(
            action_space_name=self.config.action_space_name,
            parameter_space=sample_parameter_space,
            algorithm_name=self.algorithm.algorithm_name,
        )

        sample_state_builder = create_state_builder(
            state_name=self.config.state_name,
            parameter_space=sample_parameter_space,
            action_names=list(sample_action_space.action_names),
        )

        self.action_space = sample_action_space.make_gym_space()
        self.observation_space = sample_state_builder.make_observation_space()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, InfoDict]:
        """
        Start a new graph-layout episode.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        graph, metadata_row = self._select_graph_from_options(options)

        self.current_graph = graph
        self.current_metadata_row = metadata_row

        self.parameter_space = self.algorithm.get_parameter_space(
            graph=graph,
            layout_scale=self.config.layout_scale,
        )

        self.parameter_action_space = create_action_space(
            action_space_name=self.config.action_space_name,
            parameter_space=self.parameter_space,
            algorithm_name=self.algorithm.algorithm_name,
        )

        self.state_builder = create_state_builder(
            state_name=self.config.state_name,
            parameter_space=self.parameter_space,
            action_names=list(self.parameter_action_space.action_names),
        )

        positions = self.algorithm.initialize_layout(
            graph=graph,
            seed=int(self.rng.integers(0, 2**31 - 1)),
            layout_scale=self.config.layout_scale,
        )

        parameters = self.algorithm.default_parameters(
            graph=graph,
            layout_scale=self.config.layout_scale,
        )

        initial_metrics = self.layout_evaluator.evaluate(
            graph=graph,
            positions=positions,
        )

        initial_scores = self.score_calculator.score(
            metrics=initial_metrics,
        )

        graph_id = metadata_row.get(
            self.config.graph_id_key,
            graph.graph.get("graph_id", "unknown_graph"),
        )

        self.context = LayoutContext.create_initial(
            graph=graph,
            algorithm_name=self.algorithm.algorithm_name,
            positions=positions,
            parameters=parameters,
            layout_scale=self.config.layout_scale,
            max_steps=self.config.max_macro_steps,
            metrics=initial_metrics,
            scores=initial_scores,
            layout_stats={},
            action_names=list(self.parameter_action_space.action_names),
            graph_id=str(graph_id),
        )

        observation = self._build_observation()
        info = self._build_info(
            reward_result_info={},
            action_result=None,
            terminated=False,
            truncated=False,
        )

        return observation, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, InfoDict]:
        """
        Execute one RL macro-step.
        """
        if self.context is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self.current_graph is None:
            raise RuntimeError("No active graph. Call reset() first.")

        if self.parameter_space is None:
            raise RuntimeError("Parameter space is not initialized.")

        if self.parameter_action_space is None:
            raise RuntimeError("Action space is not initialized.")

        action_id = int(action)

        action_result = self.parameter_action_space.apply(
            parameters=self.context.parameters,
            action_id=action_id,
        )

        layout_result = self.algorithm.step(
            graph=self.current_graph,
            positions=self.context.positions,
            parameters=action_result.new_parameters,
            iterations=self.config.iterations_per_step,
            layout_scale=self.config.layout_scale,
        )

        new_metrics = self.layout_evaluator.evaluate(
            graph=self.current_graph,
            positions=layout_result.positions,
        )

        new_scores = self.score_calculator.score(
            metrics=new_metrics,
        )

        self.context.update_after_step(
            action_id=action_result.action_id,
            action_name=action_result.action_name,
            new_positions=layout_result.positions,
            new_parameters=layout_result.parameters,
            new_metrics=new_metrics,
            new_scores=new_scores,
            new_layout_stats=layout_result.stats,
        )

        terminated = self.context.is_terminal

        if self.config.enable_early_stopping:
            terminated = terminated or self._check_early_stopping()

        truncated = False

        reward_result = self.reward_function.compute(
            context=self.context,
            action_result=action_result,
            is_terminal=terminated,
        )

        observation = self._build_observation()

        info = self._build_info(
            reward_result_info=reward_result.to_info_dict(),
            action_result=action_result,
            terminated=terminated,
            truncated=truncated,
        )

        return observation, float(reward_result.reward), terminated, truncated, info

    def close(self) -> None:
        """
        Gymnasium close hook.
        """
        return None

    def _build_observation(self) -> np.ndarray:
        if self.context is None:
            raise RuntimeError("Context is not initialized.")

        if self.parameter_space is None:
            raise RuntimeError("Parameter space is not initialized.")

        if self.state_builder is None:
            raise RuntimeError("State builder is not initialized.")

        observation = self.state_builder.build(
            context=self.context,
            parameter_space=self.parameter_space,
        )

        return observation

    def _load_split_metadata(
        self,
        metadata_path: str,
        split: str,
    ) -> List[MetadataRow]:
        path = Path(metadata_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                "Run `python -m graph_data.dataset_builder` first."
            )

        rows = load_metadata_csv(path)

        split_rows = [
            row for row in rows if row.get("split", "") == split
        ]

        if not split_rows:
            available_splits = sorted(set(row.get("split", "") for row in rows))
            raise ValueError(
                f"No graphs found for split='{split}'. "
                f"Available splits: {available_splits}"
            )

        return split_rows

    def _sample_graph(self) -> Tuple[nx.Graph, MetadataRow]:
        index = int(self.rng.integers(0, len(self.metadata_rows)))
        row = self.metadata_rows[index]

        graph_path = row.get(self.config.graph_path_key, "")

        if not graph_path:
            raise ValueError(
                f"Metadata row is missing '{self.config.graph_path_key}'."
            )

        graph = load_graph_pickle(graph_path)

        return graph, row

    def _select_graph_from_options(
        self,
        options: Optional[Dict[str, Any]],
    ) -> Tuple[nx.Graph, MetadataRow]:
        """
        Select graph for reset().

        Options can include:
            graph_index: int
            graph_id: str
        """
        if options is None:
            return self._sample_graph()

        if "graph_index" in options:
            graph_index = int(options["graph_index"])

            if graph_index < 0 or graph_index >= len(self.metadata_rows):
                raise ValueError(
                    f"graph_index must be in [0, {len(self.metadata_rows) - 1}], "
                    f"got {graph_index}."
                )

            row = self.metadata_rows[graph_index]
            graph = load_graph_pickle(row[self.config.graph_path_key])
            return graph, row

        if "graph_id" in options:
            target_graph_id = str(options["graph_id"])

            for row in self.metadata_rows:
                if row.get(self.config.graph_id_key, "") == target_graph_id:
                    graph = load_graph_pickle(row[self.config.graph_path_key])
                    return graph, row

            raise ValueError(f"graph_id not found in split: {target_graph_id}")

        return self._sample_graph()

    def _check_early_stopping(self) -> bool:
        """
        Optional normalized convergence check.

        Disabled by default for stable PPO training episodes.
        """
        if self.context is None:
            return False

        mean_displacement = float(
            self.context.layout_stats.get("mean_node_displacement", 0.0)
        )

        displacement_rate = abs(
            self.context.get_layout_stat_delta(
                "mean_node_displacement",
                default=0.0,
            )
        )

        mean_threshold = (
            self.config.average_displacement_threshold
            * max(1e-9, self.config.layout_scale)
        )

        rate_threshold = (
            self.config.displacement_rate_threshold
            * max(1e-9, self.config.layout_scale)
        )

        return bool(
            mean_displacement <= mean_threshold
            and displacement_rate <= rate_threshold
            and self.context.current_step > 1
        )

    def _build_info(
        self,
        reward_result_info: Dict[str, float],
        action_result: Optional[ActionResult],
        terminated: bool,
        truncated: bool,
    ) -> InfoDict:
        if self.context is None:
            return {}

        row = self.current_metadata_row if self.current_metadata_row is not None else {}

        info: InfoDict = {
            "graph_id": self.context.graph_id,
            "split": row.get("split", self.config.split),
            "family": row.get("family", "unknown"),
            "size_label": row.get("size_label", "unknown"),
            "n": self.context.graph.number_of_nodes(),
            "m": self.context.graph.number_of_edges(),
            "current_step": self.context.current_step,
            "max_macro_steps": self.config.max_macro_steps,
            "iterations_per_step": self.config.iterations_per_step,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "layout_score": float(self.context.scores.get("layout_score", 0.0)),
            "delta_layout_score": float(
                self.context.get_score_delta("layout_score", 0.0)
            ),
        }

        if action_result is not None:
            info.update(action_result.to_info_dict())

        for key, value in self.context.parameters.items():
            info[f"parameter::{key}"] = float(value)

        for key, value in self.context.layout_stats.items():
            info[f"layout_stat::{key}"] = float(value)

        info.update(reward_result_info)

        return info