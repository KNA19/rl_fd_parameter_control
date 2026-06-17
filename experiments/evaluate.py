from __future__ import annotations

from pathlib import Path

from agents import EvaluationConfig, evaluate_ppo
from graph_data import DatasetBuildConfig, build_dataset


def main() -> None:
    """
    Evaluate the trained PPO model.

    By default, this evaluates on the validation split.
    """

    metadata_path = "data/metadata/dataset_metadata.csv"
    model_path = "outputs/models/ppo/fd_param_control_ppo.zip"

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

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run `python -m experiments.train` first."
        )

    config = EvaluationConfig(
        metadata_path=metadata_path,
        split="val",
        layout_scale=1.0,
        max_macro_steps=5,
        iterations_per_step=20,
        seed=2026,
        algorithm_name="fruchterman_reingold",
        state_name="full",
        action_space_name="pure_fr_multiscale",
        reward_name="aesthetic_delta",
        model_path=model_path,
        num_episodes=10,
        deterministic=True,
        output_csv_path="outputs/evaluation/val_evaluation_summary.csv",
    )

    evaluate_ppo(config)


if __name__ == "__main__":
    main()