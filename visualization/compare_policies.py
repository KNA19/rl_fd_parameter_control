from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from envs import FDParamControlEnvConfig
from graph_data import DatasetBuildConfig, build_dataset
from visualization.layout_plotter import plot_policy_comparison
from visualization.layout_runner import PolicyLayoutRunResult, run_policy_on_graph


@dataclass(frozen=True)
class VisualComparisonConfig:
    """
    Configuration for Step 13 visual comparison.
    """

    metadata_path: str = "data/metadata/dataset_metadata.csv"
    split: str = "val"

    layout_scale: float = 1.0
    max_macro_steps: int = 5
    iterations_per_step: int = 20
    seed: int = 2026

    algorithm_name: str = "fruchterman_reingold"
    state_name: str = "full"
    action_space_name: str = "pure_fr_multiscale"
    reward_name: str = "aesthetic_delta"

    model_path: str = "outputs/models/ppo/fd_param_control_ppo.zip"

    graph_indices: Sequence[int] = (0, 1, 2)
    output_dir: str = "outputs/visuals"
    summary_csv_path: str = "outputs/visuals/visual_comparison_summary.csv"

    include_ppo: bool = True


def run_visual_comparison(
    config: VisualComparisonConfig,
) -> List[PolicyLayoutRunResult]:
    """
    Run visual comparison for selected graphs.

    For each graph, compare:
        no_change
        random
        fixed::large_decrease_k
        ppo, if available and enabled
    """
    _ensure_dataset_exists(
        metadata_path=config.metadata_path,
    )

    model_exists = Path(config.model_path).exists()

    if config.include_ppo and not model_exists:
        print(
            f"PPO model not found at {config.model_path}. "
            "PPO panel will be skipped."
        )

    all_results: List[PolicyLayoutRunResult] = []

    env_config = FDParamControlEnvConfig(
        metadata_path=config.metadata_path,
        split=config.split,
        layout_scale=config.layout_scale,
        max_macro_steps=config.max_macro_steps,
        iterations_per_step=config.iterations_per_step,
        seed=config.seed,
        state_name=config.state_name,
        action_space_name=config.action_space_name,
        reward_name=config.reward_name,
        enable_early_stopping=False,
    )

    for graph_index in config.graph_indices:
        graph_seed = config.seed + int(graph_index)

        results_for_graph: List[PolicyLayoutRunResult] = []

        results_for_graph.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=int(graph_index),
                policy_name="no_change",
                seed=graph_seed,
                algorithm_name=config.algorithm_name,
            )
        )

        results_for_graph.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=int(graph_index),
                policy_name="random",
                seed=graph_seed,
                algorithm_name=config.algorithm_name,
            )
        )

        results_for_graph.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=int(graph_index),
                policy_name="fixed",
                fixed_action_name="large_decrease_k",
                seed=graph_seed,
                algorithm_name=config.algorithm_name,
            )
        )

        if config.include_ppo and model_exists:
            results_for_graph.append(
                run_policy_on_graph(
                    config=env_config,
                    graph_index=int(graph_index),
                    policy_name="ppo",
                    model_path=config.model_path,
                    seed=graph_seed,
                    algorithm_name=config.algorithm_name,
                )
            )

        all_results.extend(results_for_graph)

        graph_id = results_for_graph[0].graph_id
        safe_graph_id = _safe_filename(graph_id)

        output_path = (
            Path(config.output_dir)
            / f"policy_visual_comparison_graph_{graph_index}_{safe_graph_id}.png"
        )

        plot_policy_comparison(
            results=results_for_graph,
            output_path=str(output_path),
            title="Step 13 Visual Diagnostic Comparison",
        )

    save_visual_comparison_summary(
        results=all_results,
        output_csv_path=config.summary_csv_path,
    )

    print_visual_comparison_summary(all_results)

    return all_results


def save_visual_comparison_summary(
    results: List[PolicyLayoutRunResult],
    output_csv_path: str,
) -> None:
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for result in results:
        rows.append(
            {
                "policy": result.policy_name,
                "graph_id": result.graph_id,
                "family": result.family,
                "size_label": result.size_label,
                "n": result.graph.number_of_nodes(),
                "m": result.graph.number_of_edges(),
                "initial_score": result.initial_score,
                "final_score": result.final_score,
                "improvement": result.improvement,
                "total_reward": result.total_reward,
                "actions": " | ".join(result.action_sequence),
            }
        )

    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved visual comparison summary to: {output_path}")


def print_visual_comparison_summary(
    results: List[PolicyLayoutRunResult],
) -> None:
    print("\nVisual comparison summary")
    print("-------------------------")

    for result in results:
        print(
            f"{result.policy_name:28s} | "
            f"{result.family:16s} | "
            f"initial={result.initial_score:.6f} | "
            f"final={result.final_score:.6f} | "
            f"Δ={result.improvement:+.6f} | "
            f"actions={_short_action_summary(result.action_sequence)}"
        )


def _short_action_summary(
    actions: List[str],
) -> str:
    if not actions:
        return "none"

    if len(set(actions)) == 1:
        return f"{actions[0]}×{len(actions)}"

    return ", ".join(actions[:5])


def _ensure_dataset_exists(
    metadata_path: str,
) -> None:
    if Path(metadata_path).exists():
        return

    print("Dataset metadata not found. Building default dataset first...")

    build_dataset(
        config=DatasetBuildConfig(
            output_graph_dir="data/processed/graphs",
            metadata_path=metadata_path,
            base_seed=2026,
            overwrite=True,
        )
    )


def _safe_filename(
    text: str,
) -> str:
    safe = (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("|", "_")
    )

    return safe[:120]