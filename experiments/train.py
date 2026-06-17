from __future__ import annotations

from pathlib import Path

from agents import PPOTrainingConfig, train_ppo, training_result_to_dict
from graph_data import DatasetBuildConfig, build_dataset


def main() -> None:
    """
    Train the first PPO model using the final framework.

    This is a debug training setup. Increase total_timesteps and episode
    length for final experiments.
    """

    metadata_path = "data/metadata/dataset_metadata.csv"

    if not Path(metadata_path).exists():
        print("Dataset metadata not found. Building default dataset first...")
        build_dataset(
            config=DatasetBuildConfig(
                output_graph_dir="data/processed/graphs",
                metadata_path=metadata_path,
                base_seed=2026,
                overwrite=True,
            )
        )

    config = PPOTrainingConfig(
        metadata_path=metadata_path,
        split="train",
        layout_scale=1.0,
        max_macro_steps=5,
        iterations_per_step=20,
        seed=2026,
        algorithm_name="fruchterman_reingold",
        state_name="full",
        action_space_name="pure_fr_multiscale",
        reward_name="aesthetic_delta",
        total_timesteps=50000,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.02,
        vf_coef=0.50,
        model_output_path="outputs/models/ppo/fd_param_control_ppo_50k.zip",
        tensorboard_log_path=None,
        check_environment=True,
        verbose=1,
    )

    _model, result = train_ppo(config)

    print("\nTraining result:")
    print(training_result_to_dict(result))


if __name__ == "__main__":
    main()