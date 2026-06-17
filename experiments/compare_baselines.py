from __future__ import annotations

from pathlib import Path

from agents import (
    EvaluationConfig,
    evaluate_fixed_action_policy,
    evaluate_ppo,
    evaluate_random_policy,
    print_policy_comparison,
    save_comparison_csv,
)
from graph_data import DatasetBuildConfig, build_dataset


def main() -> None:
    """
    Compare PPO against simple baselines:

        no_change
        large_decrease_k
        random
        PPO

    This should be run before visual comparison in Step 13.
    """

    metadata_path = "data/metadata/dataset_metadata.csv"
    model_path = "outputs/models/ppo/fd_param_control_ppo_50k.zip"

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

    base_config = EvaluationConfig(
        metadata_path=metadata_path,
        split="test_unseen_family",
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
        output_csv_path=None,
    )

    summaries = []

    no_change_config = EvaluationConfig(
        **{
            **base_config.__dict__,
            "output_csv_path": "outputs/evaluation/no_change_test_unseen_family.csv",
        }
    )

    summaries.append(
        evaluate_fixed_action_policy(
            config=no_change_config,
            action_name="no_change",
        )
    )

    large_decrease_config = EvaluationConfig(
        **{
            **base_config.__dict__,
            "output_csv_path": "outputs/evaluation/fixed_large_decrease_k_test_unseen_family.csv",
        }
    )

    summaries.append(
        evaluate_fixed_action_policy(
            config=large_decrease_config,
            action_name="large_decrease_k",
        )
    )

    random_config = EvaluationConfig(
        **{
            **base_config.__dict__,
            "output_csv_path": "outputs/evaluation/random_test_unseen_family.csv",
        }
    )

    summaries.append(
        evaluate_random_policy(
            config=random_config,
        )
    )

    if Path(model_path).exists():
        ppo_config = EvaluationConfig(
            **{
                **base_config.__dict__,
                "output_csv_path": "outputs/evaluation/ppo_test_unseen_family.csv",
            }
        )

        summaries.append(
            evaluate_ppo(
                config=ppo_config,
            )
        )
    else:
        print(
            f"\nPPO model not found at {model_path}. "
            "Skipping PPO evaluation. Run `python -m experiments.train` first."
        )

    print_policy_comparison(summaries)

    save_comparison_csv(
        summaries=summaries,
        output_csv_path="outputs/evaluation/policy_comparison_test_unseen_family.csv",
    )


if __name__ == "__main__":
    main()