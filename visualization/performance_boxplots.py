from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from experiments.fr_experiment_settings import FR_EXPERIMENT_SETTING


@dataclass(frozen=True)
class BoxPlotRecord:
    split: str
    policy: str
    seed: int
    graph_id: str
    improvement: float


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
    Select the best fixed-action baseline for each split using the aggregate
    summary from Step 4.

    fixed::no_change is excluded because no_change/default FR is plotted
    separately.
    """
    rows = read_aggregate_summary()

    best_fixed: Dict[str, str] = {}

    for split in FR_EXPERIMENT_SETTING.evaluation_splits:
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


def load_policy_episode_records(
    seed: int,
    split: str,
    policy_name: str,
    display_name: str,
) -> List[BoxPlotRecord]:
    """
    Load per-episode evaluation rows for one seed/split/policy.

    Each row contributes one point to the box plot.
    """
    csv_path = Path(
        FR_EXPERIMENT_SETTING.evaluation_csv_path(
            seed=seed,
            split=split,
            policy_name=policy_name,
        )
    )

    rows = read_csv_rows(csv_path)

    records: List[BoxPlotRecord] = []

    for row in rows:
        improvement = float(row["layout_score_improvement"])

        records.append(
            BoxPlotRecord(
                split=split,
                policy=display_name,
                seed=seed,
                graph_id=str(row.get("graph_id", "")),
                improvement=improvement,
            )
        )

    return records


def collect_figure_set1_records() -> List[BoxPlotRecord]:
    """
    Collect all records needed for Figure Set 1.

    Policies:
        - default FR / no_change
        - random
        - best fixed-action baseline per split
        - PPO
    """
    best_fixed_by_split = get_best_fixed_action_by_split()

    all_records: List[BoxPlotRecord] = []

    for split in FR_EXPERIMENT_SETTING.evaluation_splits:
        best_fixed_action = best_fixed_by_split[split]
        best_fixed_policy_name = f"fixed::{best_fixed_action}"

        for seed in FR_EXPERIMENT_SETTING.training_seeds:
            all_records.extend(
                load_policy_episode_records(
                    seed=seed,
                    split=split,
                    policy_name="fixed::no_change",
                    display_name="Default FR",
                )
            )

            all_records.extend(
                load_policy_episode_records(
                    seed=seed,
                    split=split,
                    policy_name="random",
                    display_name="Random",
                )
            )

            all_records.extend(
                load_policy_episode_records(
                    seed=seed,
                    split=split,
                    policy_name=best_fixed_policy_name,
                    display_name="Best fixed",
                )
            )

            all_records.extend(
                load_policy_episode_records(
                    seed=seed,
                    split=split,
                    policy_name="ppo",
                    display_name="PPO",
                )
            )

    return all_records


def save_figure_set1_records(
    records: Sequence[BoxPlotRecord],
    output_csv_path: Path,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "split": record.split,
            "policy": record.policy,
            "seed": record.seed,
            "graph_id": record.graph_id,
            "layout_score_improvement": record.improvement,
        }
        for record in records
    ]

    if not rows:
        return

    with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Figure Set 1 data to: {output_csv_path}")


def plot_layout_improvement_boxplots(
    records: Sequence[BoxPlotRecord],
    output_path: Path,
) -> None:
    """
    Generate Figure Set 1:

    Box plots of layout-score improvement for:
        Default FR, Random, Best fixed, PPO

    across:
        val, test_seen, test_unseen_size, test_unseen_family
    """
    if not records:
        raise ValueError("No records provided for plotting.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = list(FR_EXPERIMENT_SETTING.evaluation_splits)

    policies = [
        "Default FR",
        "Random",
        "Best fixed",
        "PPO",
    ]

    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(18, 5.2),
        squeeze=False,
    )

    axes_row = axes[0]

    for axis, split in zip(axes_row, splits):
        _plot_one_split(
            axis=axis,
            records=records,
            split=split,
            policies=policies,
        )

    fig.suptitle(
        "Figure Set 1: Layout-score improvement by policy",
        fontsize=14,
        y=0.98,
    )

    fig.text(
        0.5,
        0.02,
        "Policy",
        ha="center",
        fontsize=11,
    )

    fig.text(
        0.01,
        0.5,
        "Layout-score improvement",
        va="center",
        rotation="vertical",
        fontsize=11,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.84,
        bottom=0.20,
        wspace=0.28,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure Set 1 box plot to: {output_path}")


def _plot_one_split(
    axis: Axes,
    records: Sequence[BoxPlotRecord],
    split: str,
    policies: Sequence[str],
) -> None:
    split_records = [
        record for record in records
        if record.split == split
    ]

    data: List[List[float]] = []

    for policy in policies:
        values = [
            record.improvement
            for record in split_records
            if record.policy == policy
        ]

        data.append(values)

    axis.boxplot(
        data,
        tick_labels=list(policies),
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
    )

    axis.set_title(_pretty_split_name(split), fontsize=11)
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", labelrotation=25)

    means = [
        float(np.mean(values)) if values else float("nan")
        for values in data
    ]

    ppo_mean = means[-1]

    axis.text(
        0.5,
        1.02,
        f"PPO mean Δ={ppo_mean:.4f}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )


def _pretty_split_name(split: str) -> str:
    mapping = {
        "val": "Validation",
        "test_seen": "Test seen",
        "test_unseen_size": "Test unseen size",
        "test_unseen_family": "Test unseen family",
    }

    return mapping.get(split, split)


def generate_figure_set1() -> None:
    records = collect_figure_set1_records()

    output_dir = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "figures"
        / "figure_set_1"
    )

    save_figure_set1_records(
        records=records,
        output_csv_path=output_dir / "figure_set_1_layout_improvement_data.csv",
    )

    plot_layout_improvement_boxplots(
        records=records,
        output_path=output_dir / "figure_set_1_layout_improvement_boxplots.png",
    )

    print_summary(records)


def print_summary(records: Sequence[BoxPlotRecord]) -> None:
    print("\nFigure Set 1 summary")
    print("--------------------")

    splits = list(FR_EXPERIMENT_SETTING.evaluation_splits)

    policies = [
        "Default FR",
        "Random",
        "Best fixed",
        "PPO",
    ]

    for split in splits:
        print(f"\nSplit: {split}")

        for policy in policies:
            values = [
                record.improvement
                for record in records
                if record.split == split and record.policy == policy
            ]

            if not values:
                print(f"  {policy:12s}: missing")
                continue

            print(
                f"  {policy:12s}: "
                f"mean={np.mean(values):.6f}, "
                f"std={np.std(values, ddof=0):.6f}, "
                f"n={len(values)}"
            )