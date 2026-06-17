import numpy as np

from agents import (
    EvaluationConfig,
    evaluate_fixed_action_policy,
    evaluate_random_policy,
)
from graph_data import DatasetBuildConfig, SplitSpec, build_dataset


def main() -> None:
    """
    Test baseline evaluation before Step 13.

    This does not require a trained PPO model.
    """

    test_graph_dir = "data/processed/test_baseline_graphs"
    test_metadata_path = "data/metadata/test_baseline_dataset_metadata.csv"

    split_specs = (
        SplitSpec(
            split_name="val",
            families=("erdos_renyi", "tree"),
            size_labels=("small",),
            graphs_per_family_size=2,
            seed_offset=1000,
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

    base_config = EvaluationConfig(
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
        model_path="outputs/models/ppo/not_needed_for_baseline.zip",
        num_episodes=2,
        deterministic=True,
        output_csv_path=None,
    )

    no_change_summary = evaluate_fixed_action_policy(
        config=base_config,
        action_name="no_change",
    )

    large_decrease_summary = evaluate_fixed_action_policy(
        config=base_config,
        action_name="large_decrease_k",
    )

    random_summary = evaluate_random_policy(
        config=base_config,
    )

    for summary in [
        no_change_summary,
        large_decrease_summary,
        random_summary,
    ]:
        assert summary["num_episodes"] == 2
        assert np.isfinite(summary["mean_total_reward"])
        assert np.isfinite(summary["mean_initial_layout_score"])
        assert np.isfinite(summary["mean_final_layout_score"])
        assert np.isfinite(summary["mean_layout_score_improvement"])

    print("\nBaseline evaluation test passed.")
    print(
        "No-change improvement: "
        f"{no_change_summary['mean_layout_score_improvement']:.6f}"
    )
    print(
        "Fixed large_decrease_k improvement: "
        f"{large_decrease_summary['mean_layout_score_improvement']:.6f}"
    )
    print(
        "Random improvement: "
        f"{random_summary['mean_layout_score_improvement']:.6f}"
    )


if __name__ == "__main__":
    main()