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
class CombinedTestFinalScoreRecord:
    split: str
    policy: str
    seed: int
    graph_id: str
    final_score: float


TEST_SPLITS = (
    "test_seen",
    "test_unseen_size",
    "test_unseen_family",
)

POLICIES = (
    "Default FR",
    "Random",
    "Best fixed",
    "PPO",
)

BOX_COLORS = (
    "#8ecae6",  # Default FR
    "#ffb703",  # Random
    "#bdb2ff",  # Best fixed
    "#90be6d",  # PPO
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

    The selection is based on mean layout-score improvement, so Figure 2 uses
    the same best-fixed baseline choice as Figure 1.

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


def load_policy_episode_records(
    seed: int,
    split: str,
    policy_name: str,
    display_name: str,
) -> List[CombinedTestFinalScoreRecord]:
    """
    Load episode-level final layout scores for one seed, split, and policy.
    """
    csv_path = Path(
        FR_EXPERIMENT_SETTING.evaluation_csv_path(
            seed=seed,
            split=split,
            policy_name=policy_name,
        )
    )

    rows = read_csv_rows(csv_path)

    records: List[CombinedTestFinalScoreRecord] = []

    for row in rows:
        records.append(
            CombinedTestFinalScoreRecord(
                split=split,
                policy=display_name,
                seed=seed,
                graph_id=str(row.get("graph_id", "")),
                final_score=float(row["final_layout_score"]),
            )
        )

    return records


def collect_combined_test_final_score_records() -> List[CombinedTestFinalScoreRecord]:
    """
    Collect final layout score records from all test splits together.

    Combined test class:
        test_seen
        test_unseen_size
        test_unseen_family

    Policies:
        Default FR
        Random
        Best fixed
        PPO
    """
    best_fixed_by_split = get_best_fixed_action_by_split()

    records: List[CombinedTestFinalScoreRecord] = []

    for split in TEST_SPLITS:
        best_fixed_action = best_fixed_by_split[split]
        best_fixed_policy_name = f"fixed::{best_fixed_action}"

        for seed in FR_EXPERIMENT_SETTING.training_seeds:
            records.extend(
                load_policy_episode_records(
                    seed=seed,
                    split=split,
                    policy_name="fixed::no_change",
                    display_name="Default FR",
                )
            )

            records.extend(
                load_policy_episode_records(
                    seed=seed,
                    split=split,
                    policy_name="random",
                    display_name="Random",
                )
            )

            records.extend(
                load_policy_episode_records(
                    seed=seed,
                    split=split,
                    policy_name=best_fixed_policy_name,
                    display_name="Best fixed",
                )
            )

            records.extend(
                load_policy_episode_records(
                    seed=seed,
                    split=split,
                    policy_name="ppo",
                    display_name="PPO",
                )
            )

    return records


def save_combined_test_final_score_data(
    records: Sequence[CombinedTestFinalScoreRecord],
    output_csv_path: Path,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "split": record.split,
            "policy": record.policy,
            "seed": record.seed,
            "graph_id": record.graph_id,
            "final_layout_score": record.final_score,
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

    print(f"Saved combined test final-score data to: {output_csv_path}")


def plot_combined_test_final_score_boxplot(
    records: Sequence[CombinedTestFinalScoreRecord],
    output_path: Path,
) -> None:
    """
    Generate Figure 2:

    One combined test-class box plot for final layout score.

    No title.
    No footer.
    Colored boxes.
    """
    if not records:
        raise ValueError("No records provided for plotting.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(
        1,
        1,
        figsize=(5.8, 4.2),
    )

    _plot_single_combined_final_score_boxplot(
        axis=axis,
        records=records,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.98,
        bottom=0.18,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 2 combined test final-score box plot to: {output_path}")


def _plot_single_combined_final_score_boxplot(
    axis: Axes,
    records: Sequence[CombinedTestFinalScoreRecord],
) -> None:
    data: List[List[float]] = []

    for policy in POLICIES:
        values = [
            record.final_score
            for record in records
            if record.policy == policy
        ]

        data.append(values)

    boxplot = axis.boxplot(
        data,
        tick_labels=list(POLICIES),
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        medianprops={
            "color": "black",
            "linewidth": 1.2,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 0.8,
        },
        capprops={
            "color": "black",
            "linewidth": 0.8,
        },
    )

    for patch, color in zip(boxplot["boxes"], BOX_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    axis.set_title("")
    axis.set_xlabel("")
    axis.set_ylabel("Final layout score", fontsize=10)

    axis.grid(axis="y", alpha=0.30)
    axis.tick_params(axis="x", labelrotation=20, labelsize=9)
    axis.tick_params(axis="y", labelsize=9)


def generate_figure2_combined_test_final_score() -> None:
    records = collect_combined_test_final_score_records()

    output_dir = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "figures"
        / "figure_2_combined_test"
    )

    save_combined_test_final_score_data(
        records=records,
        output_csv_path=output_dir / "figure_2_combined_test_final_score_data.csv",
    )

    plot_combined_test_final_score_boxplot(
        records=records,
        output_path=output_dir / "figure_2_combined_test_final_score_boxplot.png",
    )

    print_summary(records)


def print_summary(records: Sequence[CombinedTestFinalScoreRecord]) -> None:
    print("\nFigure 2 combined test final-score summary")
    print("------------------------------------------")

    for policy in POLICIES:
        values = [
            record.final_score
            for record in records
            if record.policy == policy
        ]

        if not values:
            print(f"{policy:12s}: missing")
            continue

        print(
            f"{policy:12s}: "
            f"mean={np.mean(values):.6f}, "
            f"std={np.std(values, ddof=0):.6f}, "
            f"n={len(values)}"
        )