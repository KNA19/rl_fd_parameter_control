from pathlib import Path

from graph_data import DatasetBuildConfig, SplitSpec, build_dataset
from visualization import VisualComparisonConfig, run_visual_comparison


def main() -> None:
    """
    Step 13 test.

    This test does not require a PPO model. It checks that visual comparison
    works for no_change, random, and fixed large_decrease_k.
    """

    test_graph_dir = "data/processed/test_visual_graphs"
    test_metadata_path = "data/metadata/test_visual_dataset_metadata.csv"
    output_dir = "outputs/visuals/test"
    summary_csv_path = "outputs/visuals/test/visual_comparison_summary.csv"

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

    config = VisualComparisonConfig(
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
        model_path="outputs/models/ppo/not_needed_for_test.zip",
        graph_indices=(0,),
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        include_ppo=False,
    )

    results = run_visual_comparison(config)

    assert len(results) == 3
    assert Path(summary_csv_path).exists()

    png_files = list(Path(output_dir).glob("*.png"))

    assert len(png_files) >= 1

    print("\nStep 13 visual comparison test passed.")
    print(f"Generated image: {png_files[0]}")
    print(f"Generated summary: {summary_csv_path}")


if __name__ == "__main__":
    main()