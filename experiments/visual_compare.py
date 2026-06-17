from __future__ import annotations

from visualization import VisualComparisonConfig, run_visual_comparison


def main() -> None:
    """
    Step 13 visual comparison entry point.

    Run:
        python -m experiments.visual_compare

    Output:
        outputs/visuals/*.png
        outputs/visuals/visual_comparison_summary.csv
    """

    config = VisualComparisonConfig(
        metadata_path="data/metadata/dataset_metadata.csv",
        split="val",
        layout_scale=1.0,
        max_macro_steps=5,
        iterations_per_step=20,
        seed=2026,
        algorithm_name="fruchterman_reingold",
        state_name="full",
        action_space_name="pure_fr_multiscale",
        reward_name="aesthetic_delta",
        model_path="outputs/models/ppo/fd_param_control_ppo_50k.zip",
        graph_indices=(0, 1, 2),
        output_dir="outputs/visuals",
        summary_csv_path="outputs/visuals/visual_comparison_summary.csv",
        include_ppo=True,
    )

    run_visual_comparison(config)


if __name__ == "__main__":
    main()