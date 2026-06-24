from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from agents.evaluator import EvaluationConfig
from agents.ppo_trainer import PPOTrainingConfig
from visualization import VisualComparisonConfig


@dataclass(frozen=True)
class FRExperimentSetting:
    """
    Central configuration for the FR-only actual experiments.

    This file freezes the settings used for the publishable FR-only
    experimental phase.

    Main design:
        Algorithm: Pure Fruchterman-Reingold
        Agent: PPO
        State: full SARL state
        Action space: pure_fr_multiscale
        Reward: aesthetic_delta
    """

    experiment_name: str = "fr_sarl_v1"

    metadata_path: str = "data/metadata/dataset_metadata.csv"

    algorithm_name: str = "fruchterman_reingold"
    state_name: str = "full"
    action_space_name: str = "pure_fr_multiscale"
    reward_name: str = "aesthetic_delta"

    layout_scale: float = 1.0

    # Episode setting.
    # Total FD iterations per episode = max_macro_steps * iterations_per_step.
    # Here: 5 * 20 = 100 FD iterations per episode.
    max_macro_steps: int = 5
    iterations_per_step: int = 20

    # PPO training setting.
    total_timesteps: int = 100_000

    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.02
    vf_coef: float = 0.50

    # Publishable standard for this phase:
    # 5 independent training seeds.
    training_seeds: Tuple[int, ...] = (
        2026,
        2027,
        2028,
        2029,
        2030,
    )

    evaluation_splits: Tuple[str, ...] = (
        "val",
        "test_seen",
        "test_unseen_size",
        "test_unseen_family",
    )

    # Evaluation episodes per split per seed.
    evaluation_episodes: int = 30

    # Baselines:
    # 1. no_change/default FR
    # 2. random parameter-control policy
    # 3. fixed-action policies for all individual action names below
    #
    # Note:
    # no_change is included here because it is both the default FR baseline
    # and one of the fixed-action policies.
    fixed_action_names: Tuple[str, ...] = (
        "no_change",
        "small_increase_k",
        "small_decrease_k",
        "large_increase_k",
        "large_decrease_k",
        "small_increase_temperature",
        "small_decrease_temperature",
        "large_increase_temperature",
        "large_decrease_temperature",
        "small_increase_cooling_rate",
        "small_decrease_cooling_rate",
        "large_increase_cooling_rate",
        "large_decrease_cooling_rate",
        "reset_k_to_default",
        "reset_temperature_to_default",
        "reset_cooling_rate_to_default",
        "expand_and_explore",
        "expand_and_stabilize",
        "compress_and_stabilize",
        "reheat_layout",
        "cool_down_layout",
    )

    # Main baselines to highlight in paper tables.
    # The full fixed-action sweep is still saved separately.
    highlighted_baselines: Tuple[str, ...] = (
        "fixed::no_change",
        "random",
        "fixed::large_decrease_k",
        "ppo",
    )

    output_root: str = "outputs/experiments/fr_sarl_v1"

    def total_fd_iterations_per_episode(self) -> int:
        """
        Return the number of force-directed iterations per episode.
        """
        return self.max_macro_steps * self.iterations_per_step

    def model_path_for_seed(
        self,
        seed: int,
    ) -> str:
        """
        Return PPO model path for one training seed.
        """
        return str(
            Path(self.output_root)
            / "models"
            / f"fr_ppo_seed_{seed}.zip"
        )

    def evaluation_csv_path(
        self,
        seed: int,
        split: str,
        policy_name: str,
    ) -> str:
        """
        Return per-policy evaluation CSV path.
        """
        safe_policy = self._safe_name(policy_name)

        return str(
            Path(self.output_root)
            / "evaluation"
            / f"seed_{seed}"
            / f"{split}_{safe_policy}.csv"
        )

    def comparison_csv_path(
        self,
        seed: int,
        split: str,
    ) -> str:
        """
        Return one-row-per-policy comparison CSV path for one seed/split.
        """
        return str(
            Path(self.output_root)
            / "evaluation"
            / f"seed_{seed}"
            / f"{split}_policy_comparison.csv"
        )

    def fixed_action_sweep_csv_path(
        self,
        seed: int,
        split: str,
    ) -> str:
        """
        Return CSV path for all fixed-action policy results.
        """
        return str(
            Path(self.output_root)
            / "evaluation"
            / f"seed_{seed}"
            / f"{split}_fixed_action_sweep.csv"
        )

    def aggregate_csv_path(self) -> str:
        """
        Return aggregate summary CSV path across seeds and splits.
        """
        return str(
            Path(self.output_root)
            / "aggregate"
            / "fr_sarl_v1_aggregate_summary.csv"
        )

    def aggregate_fixed_action_csv_path(self) -> str:
        """
        Return aggregate fixed-action sweep CSV path.
        """
        return str(
            Path(self.output_root)
            / "aggregate"
            / "fr_sarl_v1_fixed_action_sweep_summary.csv"
        )

    def visual_output_dir(
        self,
        seed: int,
        split: str,
    ) -> str:
        """
        Return output directory for visual comparisons.
        """
        return str(
            Path(self.output_root)
            / "visuals"
            / f"seed_{seed}"
            / split
        )

    def visual_summary_csv_path(
        self,
        seed: int,
        split: str,
    ) -> str:
        """
        Return visual comparison summary CSV path.
        """
        return str(
            Path(self.output_root)
            / "visuals"
            / f"seed_{seed}"
            / split
            / "visual_comparison_summary.csv"
        )

    def make_training_config(
        self,
        seed: int,
    ) -> PPOTrainingConfig:
        """
        Create PPO training config for one seed.
        """
        return PPOTrainingConfig(
            metadata_path=self.metadata_path,
            split="train",
            layout_scale=self.layout_scale,
            max_macro_steps=self.max_macro_steps,
            iterations_per_step=self.iterations_per_step,
            seed=seed,
            algorithm_name=self.algorithm_name,
            state_name=self.state_name,
            action_space_name=self.action_space_name,
            reward_name=self.reward_name,
            total_timesteps=self.total_timesteps,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            model_output_path=self.model_path_for_seed(seed),
            tensorboard_log_path=None,
            check_environment=True,
            verbose=1,
        )

    def make_evaluation_config(
        self,
        seed: int,
        split: str,
        policy_name: str,
    ) -> EvaluationConfig:
        """
        Create evaluation config for one seed, split, and policy.
        """
        return EvaluationConfig(
            metadata_path=self.metadata_path,
            split=split,
            layout_scale=self.layout_scale,
            max_macro_steps=self.max_macro_steps,
            iterations_per_step=self.iterations_per_step,
            seed=seed,
            algorithm_name=self.algorithm_name,
            state_name=self.state_name,
            action_space_name=self.action_space_name,
            reward_name=self.reward_name,
            model_path=self.model_path_for_seed(seed),
            num_episodes=self.evaluation_episodes,
            deterministic=True,
            output_csv_path=self.evaluation_csv_path(
                seed=seed,
                split=split,
                policy_name=policy_name,
            ),
        )

    def make_visual_config(
        self,
        seed: int,
        split: str,
        graph_indices: Tuple[int, ...] = (0, 1, 2),
    ) -> VisualComparisonConfig:
        """
        Create visual comparison config.
        """
        return VisualComparisonConfig(
            metadata_path=self.metadata_path,
            split=split,
            layout_scale=self.layout_scale,
            max_macro_steps=self.max_macro_steps,
            iterations_per_step=self.iterations_per_step,
            seed=seed,
            algorithm_name=self.algorithm_name,
            state_name=self.state_name,
            action_space_name=self.action_space_name,
            reward_name=self.reward_name,
            model_path=self.model_path_for_seed(seed),
            graph_indices=graph_indices,
            output_dir=self.visual_output_dir(seed=seed, split=split),
            summary_csv_path=self.visual_summary_csv_path(
                seed=seed,
                split=split,
            ),
            include_ppo=True,
        )

    def describe(self) -> str:
        """
        Return readable experiment setting summary.
        """
        lines = [
            f"Experiment name: {self.experiment_name}",
            f"Algorithm: {self.algorithm_name}",
            f"State: {self.state_name}",
            f"Action space: {self.action_space_name}",
            f"Reward: {self.reward_name}",
            f"Layout scale: {self.layout_scale}",
            f"Max macro-steps: {self.max_macro_steps}",
            f"Iterations per step: {self.iterations_per_step}",
            (
                "Total FD iterations per episode: "
                f"{self.total_fd_iterations_per_episode()}"
            ),
            f"Total PPO timesteps: {self.total_timesteps}",
            f"Learning rate: {self.learning_rate}",
            f"n_steps: {self.n_steps}",
            f"batch_size: {self.batch_size}",
            f"n_epochs: {self.n_epochs}",
            f"gamma: {self.gamma}",
            f"gae_lambda: {self.gae_lambda}",
            f"clip_range: {self.clip_range}",
            f"ent_coef: {self.ent_coef}",
            f"vf_coef: {self.vf_coef}",
            f"Training seeds: {self.training_seeds}",
            f"Evaluation splits: {self.evaluation_splits}",
            f"Evaluation episodes: {self.evaluation_episodes}",
            f"Fixed-action baselines: {self.fixed_action_names}",
            f"Output root: {self.output_root}",
        ]

        return "\n".join(lines)

    def _safe_name(
        self,
        text: str,
    ) -> str:
        return (
            text.replace("::", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )


FR_EXPERIMENT_SETTING = FRExperimentSetting()