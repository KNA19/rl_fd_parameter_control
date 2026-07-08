from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from envs import FDParamControlEnvConfig
from experiments.fr_experiment_settings import FR_EXPERIMENT_SETTING
from visualization.layout_plotter import plot_policy_comparison
from visualization.layout_runner import PolicyLayoutRunResult, run_policy_on_graph


SummaryRow = Dict[str, Any]


def read_aggregate_summary() -> List[SummaryRow]:
    """
    Read aggregate summary produced in Step 4.
    """
    aggregate_path = Path(FR_EXPERIMENT_SETTING.aggregate_csv_path())

    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Aggregate summary not found: {aggregate_path}\n"
            "Run Step 4 first:\n"
            "python -m experiments.run_fr_multiseed_evaluation"
        )

    rows: List[SummaryRow] = []

    with open(aggregate_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(dict(row))

    return rows


def get_best_fixed_action_by_split() -> Dict[str, str]:
    """
    Find the best fixed-action baseline for each split.

    We exclude fixed::no_change here because no_change/default FR is already
    plotted separately.
    """
    rows = read_aggregate_summary()

    best_by_split: Dict[str, str] = {}

    for split in FR_EXPERIMENT_SETTING.evaluation_splits:
        fixed_rows = [
            row
            for row in rows
            if row.get("split") == split
            and str(row.get("policy", "")).startswith("fixed::")
            and row.get("policy") != "fixed::no_change"
        ]

        if not fixed_rows:
            raise ValueError(f"No fixed-action rows found for split={split}")

        best_row = max(
            fixed_rows,
            key=lambda row: float(row["mean_layout_score_improvement"]),
        )

        best_policy = str(best_row["policy"])
        action_name = best_policy.replace("fixed::", "")

        best_by_split[split] = action_name

    return best_by_split


def run_visual_examples_for_split(
    seed: int,
    split: str,
    graph_indices: Tuple[int, ...],
    best_fixed_action_name: str,
) -> List[PolicyLayoutRunResult]:
    """
    Generate visual comparisons for one split.
    """
    model_path = FR_EXPERIMENT_SETTING.model_path_for_seed(seed)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"PPO model not found: {model_path}\n"
            "Run Step 3 first:\n"
            "python -m experiments.run_fr_multiseed_training"
        )

    env_config = FDParamControlEnvConfig(
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

    all_results: List[PolicyLayoutRunResult] = []

    output_dir = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "visuals"
        / "final_examples"
        / f"seed_{seed}"
        / split
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for graph_index in graph_indices:
        graph_seed = seed + int(graph_index)

        results_for_graph: List[PolicyLayoutRunResult] = []

        # Default FR / no-change.
        results_for_graph.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=int(graph_index),
                policy_name="no_change",
                seed=graph_seed,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )
        )

        # Random policy.
        results_for_graph.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=int(graph_index),
                policy_name="random",
                seed=graph_seed,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )
        )

        # Best fixed-action baseline for this split.
        results_for_graph.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=int(graph_index),
                policy_name="fixed",
                fixed_action_name=best_fixed_action_name,
                seed=graph_seed,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )
        )

        # PPO policy.
        results_for_graph.append(
            run_policy_on_graph(
                config=env_config,
                graph_index=int(graph_index),
                policy_name="ppo",
                model_path=model_path,
                seed=graph_seed,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )
        )

        all_results.extend(results_for_graph)

        graph_id = results_for_graph[0].graph_id
        safe_graph_id = safe_filename(graph_id)

        output_path = (
            output_dir
            / f"fr_visual_example_{split}_graph_{graph_index}_{safe_graph_id}.png"
        )

        plot_policy_comparison(
            results=results_for_graph,
            output_path=str(output_path),
            title=(
                "FR-only SARL visual comparison\n"
                f"Split={split}, seed={seed}, "
                f"best fixed={best_fixed_action_name}"
            ),
        )

    save_visual_summary(
        results=all_results,
        output_csv_path=output_dir / "visual_examples_summary.csv",
    )

    return all_results


def save_visual_summary(
    results: List[PolicyLayoutRunResult],
    output_csv_path: Path,
) -> None:
    """
    Save visual-example result summary.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

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

    with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved visual summary to: {output_csv_path}")


def save_overall_summary(
    results: List[PolicyLayoutRunResult],
    output_csv_path: Path,
) -> None:
    """
    Save all visual-example results across splits.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for result in results:
        rows.append(
            {
                "split": result.final_info.get("split", "unknown"),
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

    print(f"Saved overall visual summary to: {output_csv_path}")


def print_visual_example_summary(
    results: List[PolicyLayoutRunResult],
) -> None:
    """
    Print compact summary.
    """
    print("\nVisual example summary")
    print("----------------------")

    for result in results:
        split = result.final_info.get("split", "unknown")

        print(
            f"{split:20s} | "
            f"{result.policy_name:30s} | "
            f"{result.family:16s} | "
            f"initial={result.initial_score:.6f} | "
            f"final={result.final_score:.6f} | "
            f"Δ={result.improvement:+.6f} | "
            f"actions={short_action_summary(result.action_sequence)}"
        )


def short_action_summary(actions: List[str]) -> str:
    if not actions:
        return "none"

    if len(set(actions)) == 1:
        return f"{actions[0]}×{len(actions)}"

    return ", ".join(actions[:5])


def safe_filename(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("|", "_")
    )[:120]


def main() -> None:
    """
    Step 7: Generate final visual examples.

    This uses:
        - default FR / no_change
        - random policy
        - best fixed-action baseline for each split
        - PPO policy

    Visual examples are generated for selected graph indices in each split.
    """
    seed = FR_EXPERIMENT_SETTING.training_seeds[0]

    # Use deterministic graph indices for paper-style examples.
    # You can increase this later, e.g., (0, 1, 2, 3, 4).
    graph_indices = (0, 1, 2)

    print("FR-only final visual example generation")
    print("--------------------------------------")
    print(f"Using PPO seed: {seed}")
    print(f"Graph indices: {graph_indices}")

    best_fixed_by_split = get_best_fixed_action_by_split()

    print("\nBest fixed action by split:")

    for split, action_name in best_fixed_by_split.items():
        print(f"  {split}: {action_name}")

    all_results: List[PolicyLayoutRunResult] = []

    for split in FR_EXPERIMENT_SETTING.evaluation_splits:
        print("\n" + "=" * 80)
        print(f"Generating visuals for split: {split}")
        print("=" * 80)

        split_results = run_visual_examples_for_split(
            seed=seed,
            split=split,
            graph_indices=graph_indices,
            best_fixed_action_name=best_fixed_by_split[split],
        )

        all_results.extend(split_results)

    overall_summary_path = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "visuals"
        / "final_examples"
        / f"seed_{seed}"
        / "fr_visual_examples_overall_summary.csv"
    )

    save_overall_summary(
        results=all_results,
        output_csv_path=overall_summary_path,
    )

    print_visual_example_summary(all_results)

    print("\nStep 7 visual example generation completed.")


if __name__ == "__main__":
    main()