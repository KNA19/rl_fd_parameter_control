from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes

from envs import FDParamControlEnvConfig
from experiments.fr_experiment_settings import FR_EXPERIMENT_SETTING
from visualization.layout_runner import PolicyLayoutRunResult, run_policy_on_graph


TEST_SPLITS = (
    "test_seen",
    "test_unseen_size",
    "test_unseen_family",
)

SPLIT_DISPLAY_NAMES = {
    "test_seen": "Test seen",
    "test_unseen_size": "Test unseen size",
    "test_unseen_family": "Test unseen family",
}

POLICY_COLUMNS = (
    "Initial",
    "Default FR",
    "Random",
    "Best fixed",
    "PPO",
)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def read_aggregate_summary() -> List[Dict[str, str]]:
    aggregate_path = Path(FR_EXPERIMENT_SETTING.aggregate_csv_path())

    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Aggregate summary not found: {aggregate_path}\n"
            "Run Step 4 first:\n"
            "python -m experiments.run_fr_multiseed_evaluation"
        )

    return read_csv_rows(aggregate_path)


def get_best_fixed_action_by_split() -> Dict[str, str]:
    """
    Find the best fixed-action baseline separately for each test split.

    fixed::no_change is excluded because Default FR is plotted separately.
    """
    rows = read_aggregate_summary()

    best_fixed: Dict[str, str] = {}

    for split in TEST_SPLITS:
        fixed_rows = [
            row
            for row in rows
            if row.get("split") == split
            and row.get("policy", "").startswith("fixed::")
            and row.get("policy") != "fixed::no_change"
        ]

        if not fixed_rows:
            raise ValueError(f"No fixed-action rows found for split={split}")

        best_row = max(
            fixed_rows,
            key=lambda row: float(row["mean_layout_score_improvement"]),
        )

        best_policy = str(best_row["policy"])
        best_fixed[split] = best_policy.replace("fixed::", "")

    return best_fixed


def make_env_config(
    seed: int,
    split: str,
) -> FDParamControlEnvConfig:
    return FDParamControlEnvConfig(
        metadata_path=FR_EXPERIMENT_SETTING.metadata_path,
        split=split,
        layout_scale=FR_EXPERIMENT_SETTING.layout_scale,
        max_macro_steps=FR_EXPERIMENT_SETTING.max_macro_steps,
        iterations_per_step=FR_EXPERIMENT_SETTING.iterations_per_step,
        seed=seed,
        state_name=FR_EXPERIMENT_SETTING.state_name,
        action_space_name=FR_EXPERIMENT_SETTING.action_space_name,
        reward_name=FR_EXPERIMENT_SETTING.reward_name,
        enable_early_stopping=False,
    )


def collect_visual_grid_results(
    seed: int,
    graph_index: int,
) -> Dict[str, List[PolicyLayoutRunResult]]:
    """
    Collect visual comparison results for one graph from each test split.

    For each split, run:
        Default FR
        Random
        Best fixed action
        PPO
    """
    model_path = FR_EXPERIMENT_SETTING.model_path_for_seed(seed)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"PPO model not found: {model_path}\n"
            "Run Step 3 first:\n"
            "python -m experiments.run_fr_multiseed_training"
        )

    best_fixed_by_split = get_best_fixed_action_by_split()

    split_results: Dict[str, List[PolicyLayoutRunResult]] = {}

    for split in TEST_SPLITS:
        env_config = make_env_config(
            seed=seed,
            split=split,
        )

        graph_seed = seed + graph_index
        best_fixed_action = best_fixed_by_split[split]

        results_for_split: List[PolicyLayoutRunResult] = []

        results_for_split.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=graph_index,
                policy_name="no_change",
                seed=graph_seed,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )
        )

        results_for_split.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=graph_index,
                policy_name="random",
                seed=graph_seed,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )
        )

        results_for_split.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=graph_index,
                policy_name="fixed",
                fixed_action_name=best_fixed_action,
                seed=graph_seed,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )
        )

        results_for_split.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=graph_index,
                policy_name="ppo",
                model_path=model_path,
                seed=graph_seed,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )
        )

        split_results[split] = results_for_split

    return split_results


def save_visual_grid_summary(
    split_results: Dict[str, List[PolicyLayoutRunResult]],
    output_csv_path: Path,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for split, results in split_results.items():
        for result in results:
            rows.append(
                {
                    "split": split,
                    "policy": result.policy_name,
                    "graph_id": result.graph_id,
                    "family": result.family,
                    "size_label": result.size_label,
                    "n": result.graph.number_of_nodes(),
                    "m": result.graph.number_of_edges(),
                    "initial_score": result.initial_score,
                    "final_score": result.final_score,
                    "improvement": result.improvement,
                    "actions": " | ".join(result.action_sequence),
                }
            )

    if not rows:
        return

    with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Figure 4 visual-grid summary to: {output_csv_path}")


def plot_visual_grid(
    split_results: Dict[str, List[PolicyLayoutRunResult]],
    output_path: Path,
) -> None:
    """
    Generate Figure 4:

    One visual grid using all combined test splits.

    Rows:
        test_seen
        test_unseen_size
        test_unseen_family

    Columns:
        Initial
        Default FR
        Random
        Best fixed
        PPO
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(TEST_SPLITS),
        len(POLICY_COLUMNS),
        figsize=(15.5, 9.0),
        squeeze=False,
    )

    for col_index, column_name in enumerate(POLICY_COLUMNS):
        axes[0][col_index].set_title(
            column_name,
            fontsize=10,
            pad=8,
        )

    for row_index, split in enumerate(TEST_SPLITS):
        results = split_results[split]

        if len(results) != 4:
            raise ValueError(
                f"Expected 4 policy results for split={split}, got {len(results)}."
            )

        reference = results[0]

        # Row label.
        axes[row_index][0].text(
            -0.18,
            0.5,
            SPLIT_DISPLAY_NAMES.get(split, split),
            transform=axes[row_index][0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
        )

        # Column 0: Initial layout.
        _draw_graph_layout(
            axis=axes[row_index][0],
            graph=reference.graph,
            positions=reference.initial_positions,
        )

        _add_score_label(
            axis=axes[row_index][0],
            text=f"{reference.initial_score:.3f}",
        )

        # Column 1-4: final policy layouts.
        for col_index, result in enumerate(results, start=1):
            _draw_graph_layout(
                axis=axes[row_index][col_index],
                graph=result.graph,
                positions=result.final_positions,
            )

            _add_score_label(
                axis=axes[row_index][col_index],
                text=f"{result.final_score:.3f}\nΔ={result.improvement:+.3f}",
            )

    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        top=0.94,
        bottom=0.03,
        wspace=0.08,
        hspace=0.10,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 4 visual grid to: {output_path}")


def _draw_graph_layout(
    axis: Axes,
    graph: nx.Graph,
    positions: Mapping[object, np.ndarray],
) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    pos_2d = {
        node: (
            float(np.asarray(position)[0]),
            float(np.asarray(position)[1]),
        )
        for node, position in positions.items()
    }

    nx.draw_networkx_edges(
        graph,
        pos=pos_2d,
        ax=axis,
        width=0.65,
        alpha=0.65,
    )

    nx.draw_networkx_nodes(
        graph,
        pos=pos_2d,
        ax=axis,
        node_size=28,
        linewidths=0.30,
        edgecolors="black",
    )

    _set_equal_axis_limits(
        axis=axis,
        positions=pos_2d,
    )


def _add_score_label(
    axis: Axes,
    text: str,
) -> None:
    axis.text(
        0.02,
        0.98,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.4,
            "alpha": 0.85,
        },
    )


def _set_equal_axis_limits(
    axis: Axes,
    positions: Mapping[object, tuple[float, float]],
) -> None:
    if not positions:
        return

    xs = [value[0] for value in positions.values()]
    ys = [value[1] for value in positions.values()]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)

    span = max(x_span, y_span)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    padding = 0.18 * span

    axis.set_xlim(
        x_center - 0.5 * span - padding,
        x_center + 0.5 * span + padding,
    )

    axis.set_ylim(
        y_center - 0.5 * span - padding,
        y_center + 0.5 * span + padding,
    )


def generate_figure4_combined_test_visual_grid() -> None:
    """
    Generate Figure 4 visual grid.

    Uses one representative graph index from each test split.
    """
    seed = FR_EXPERIMENT_SETTING.training_seeds[0]

    # You can change this to 1 or 2 if a different graph looks better.
    graph_index = 6

    split_results = collect_visual_grid_results(
        seed=seed,
        graph_index=graph_index,
    )

    output_dir = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "figures"
        / "figure_4_combined_test_visual_grid"
    )

    save_visual_grid_summary(
        split_results=split_results,
        output_csv_path=output_dir / "figure_4_combined_test_visual_grid_data.csv",
    )

    plot_visual_grid(
        split_results=split_results,
        output_path=output_dir / "figure_4_combined_test_visual_grid.png",
    )

    print_summary(split_results)


def print_summary(
    split_results: Dict[str, List[PolicyLayoutRunResult]],
) -> None:
    print("\nFigure 4 visual-grid summary")
    print("----------------------------")

    for split, results in split_results.items():
        print(f"\nSplit: {split}")

        for result in results:
            print(
                f"  {result.policy_name:30s} | "
                f"initial={result.initial_score:.6f} | "
                f"final={result.final_score:.6f} | "
                f"Δ={result.improvement:+.6f} | "
                f"actions={short_action_summary(result.action_sequence)}"
            )


def short_action_summary(actions: Sequence[str]) -> str:
    if not actions:
        return "none"

    if len(set(actions)) == 1:
        return f"{actions[0]}×{len(actions)}"

    return ", ".join(actions[:5])