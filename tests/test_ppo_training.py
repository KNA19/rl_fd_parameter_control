from pathlib import Path

import numpy as np

from agents import EvaluationConfig, PPOTrainingConfig, evaluate_ppo, train_ppo
from graph_data import DatasetBuildConfig, SplitSpec, build_dataset


def main() -> None:
    """
    Step 12 test.

    This performs a short PPO training run only to verify that the final
    training pipeline works.

    It is not meant to produce a strong model.
    """

    test_graph_dir = "data/processed/test_ppo_graphs"
    test_metadata_path = "data/metadata/test_ppo_dataset_metadata.csv"
    test_model_path = "outputs/models/ppo/test_fd_param_control_ppo.zip"

    split_specs = (
        SplitSpec(
            split_name="train",
            families=("erdos_renyi", "tree"),
            size_labels=("small",),
            graphs_per_family_size=2,
            seed_offset=1000,
        ),
        SplitSpec(
            split_name="val",
            families=("erdos_renyi", "tree"),
            size_labels=("small",),
            graphs_per_family_size=1,
            seed_offset=2000,
        ),
    )

    build_dataset(
        config=DatasetBuildConfig(
            output_graph_dir=test_graph_dir,
            metadata_path=test_metadata_path,
            base_seed=123,
            overwrite=True,
        ),
        split_specs=split_specs,
    )

    train_config = PPOTrainingConfig(
        metadata_path=test_metadata_path,
        split="train",
        layout_scale=1.0,
        max_macro_steps=3,
        iterations_per_step=5,
        seed=123,
        algorithm_name="fruchterman_reingold",
        state_name="full",
        action_space_name="pure_fr_multiscale",
        reward_name="aesthetic_delta",
        total_timesteps=128,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=32,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.01,
        vf_coef=0.50,
        model_output_path=test_model_path,
        tensorboard_log_path=None,
        check_environment=True,
        verbose=0,
    )

    _model, training_result = train_ppo(train_config)

    assert Path(training_result.model_path).exists()
    assert training_result.total_timesteps == train_config.total_timesteps
    assert training_result.observation_dim > 0
    assert training_result.num_actions > 0

    eval_config = EvaluationConfig(
        metadata_path=test_metadata_path,
        split="val",
        layout_scale=1.0,
        max_macro_steps=3,
        iterations_per_step=5,
        seed=123,
        algorithm_name="fruchterman_reingold",
        state_name="full",
        action_space_name="pure_fr_multiscale",
        reward_name="aesthetic_delta",
        model_path=test_model_path,
        num_episodes=2,
        deterministic=True,
        output_csv_path="outputs/evaluation/test_ppo_eval_summary.csv",
    )

    summary = evaluate_ppo(eval_config)

    assert summary["num_episodes"] == 2
    assert np.isfinite(summary["mean_total_reward"])
    assert np.isfinite(summary["mean_initial_layout_score"])
    assert np.isfinite(summary["mean_final_layout_score"])
    assert np.isfinite(summary["mean_layout_score_improvement"])

    print("\nStep 12 PPO training test passed.")
    print(f"Model path: {training_result.model_path}")
    print(f"Observation dimension: {training_result.observation_dim}")
    print(f"Number of actions: {training_result.num_actions}")
    print(f"Mean validation improvement: {summary['mean_layout_score_improvement']:.6f}")


if __name__ == "__main__":
    main()