from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.fr_experiment_settings import FR_EXPERIMENT_SETTING


SOURCE_FILENAME = "fr_sarl_v1_ppo_action_distribution_by_split.csv"

SOURCE_SPLITS = (
    "test_seen",
    "test_unseen_size",
    "test_unseen_family",
)

DISPLAY_SPLIT_NAMES = {
    "test_seen": "Seen test",
    "test_unseen_size": "Unseen size",
    "test_unseen_family": "Unseen family",
}

DISPLAY_CATEGORIES = (
    "large_increase_k",
    "no_change",
    "reset_cooling_rate_to_default",
    "reset_temperature_to_default",
    "small_decrease_k",
    "small_increase_cooling_rate",
    "Other",
)

DISPLAY_CATEGORY_NAMES = {
    "large_increase_k": "Large increase in k",
    "no_change": "No change",
    "reset_cooling_rate_to_default": "Reset cooling rate",
    "reset_temperature_to_default": "Reset temperature",
    "small_decrease_k": "Small decrease in k",
    "small_increase_cooling_rate": "Small increase in cooling rate",
    "Other": "Other",
}

CATEGORY_COLORS = {
    "large_increase_k": "#4E79A7",
    "no_change": "#F28E2B",
    "reset_cooling_rate_to_default": "#E15759",
    "reset_temperature_to_default": "#76B7B2",
    "small_decrease_k": "#59A14F",
    "small_increase_cooling_rate": "#EDC948",
    "Other": "#B07AA1",
}


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def get_source_csv_path() -> Path:
    return (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "aggregate"
        / SOURCE_FILENAME
    )


def load_action_distribution_rows() -> List[Dict[str, str]]:
    csv_path = get_source_csv_path()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Required source file not found: {csv_path}\n"
            "This figure depends on Step 6 output.\n"
            "Run:\n"
            "python -m experiments.analyze_fr_action_distributions"
        )

    return read_csv_rows(csv_path)


def aggregate_display_distribution() -> Dict[str, Dict[str, float]]:
    """
    Convert raw per-split PPO action percentages into the seven thesis categories.

    Categories:
        large_increase_k
        no_change
        reset_cooling_rate_to_default
        reset_temperature_to_default
        small_decrease_k
        small_increase_cooling_rate
        Other

    Values are percentages and sum to 100 for each split.
    """
    rows = load_action_distribution_rows()

    distribution: Dict[str, Dict[str, float]] = {}

    for split in SOURCE_SPLITS:
        split_rows = [
            row for row in rows
            if row.get("split") == split
        ]

        if not split_rows:
            raise ValueError(f"No action-distribution rows found for split={split}")

        category_to_percent: Dict[str, float] = {
            category: 0.0
            for category in DISPLAY_CATEGORIES
        }

        for row in split_rows:
            action_name = str(row.get("action_name", "")).strip()

            if not action_name:
                continue

            percentage_text = row.get("percentage", "0.0")
            percentage = float(percentage_text)

            if action_name in DISPLAY_CATEGORIES and action_name != "Other":
                category_to_percent[action_name] += percentage
            else:
                category_to_percent["Other"] += percentage

        total = sum(category_to_percent.values())

        if total > 0.0:
            scale = 100.0 / total

            for category in DISPLAY_CATEGORIES:
                category_to_percent[category] *= scale

        distribution[split] = category_to_percent

    return distribution


def save_aggregated_distribution_csv(
    distribution: Dict[str, Dict[str, float]],
    output_csv_path: Path,
) -> None:
    """
    Save the grouped action percentages used in the figure.

    This version avoids row.update(...) so Pylance does not complain about
    mixing str and float values.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["split"] + list(DISPLAY_CATEGORIES)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for split in SOURCE_SPLITS:
            row: Dict[str, str] = {
                "split": split,
            }

            for category in DISPLAY_CATEGORIES:
                row[category] = f"{distribution[split][category]:.6f}"

            writer.writerow(row)

    print(f"Saved aggregated action-distribution CSV to: {output_csv_path}")


def plot_action_distribution_figure(
    distribution: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Create a 100% stacked horizontal bar chart for the three test splits.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = list(SOURCE_SPLITS)
    y_labels = [
        DISPLAY_SPLIT_NAMES[split]
        for split in splits
    ]
    y_positions = np.arange(len(splits))

    fig, ax = plt.subplots(
        figsize=(8.8, 2.9),
    )

    left = np.zeros(len(splits), dtype=float)

    for category in DISPLAY_CATEGORIES:
        values = np.array(
            [
                distribution[split][category]
                for split in splits
            ],
            dtype=float,
        )

        bars = ax.barh(
            y_positions,
            values,
            left=left,
            color=CATEGORY_COLORS[category],
            edgecolor="white",
            linewidth=0.8,
            label=DISPLAY_CATEGORY_NAMES[category],
            height=0.58,
        )

        for bar, value in zip(bars, values):
            if value >= 7.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

        left += values

    ax.set_xlim(0.0, 100.0)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Percentage of PPO decisions", fontsize=10)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    ax.invert_yaxis()

    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        frameon=False,
        fontsize=8,
        columnspacing=1.0,
        handlelength=1.6,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.99,
        top=0.78,
        bottom=0.20,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved action-distribution figure to: {output_path}")


def generate_fr_sarl_action_distribution_figure() -> None:
    distribution = aggregate_display_distribution()

    output_dir = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "figures"
        / "fr_sarl_action_distribution"
    )

    save_aggregated_distribution_csv(
        distribution=distribution,
        output_csv_path=output_dir / "fr_sarl_ppo_action_distribution_data.csv",
    )

    plot_action_distribution_figure(
        distribution=distribution,
        output_path=output_dir / "fr_sarl_ppo_action_distribution.png",
    )

    print_summary(distribution)


def print_summary(distribution: Dict[str, Dict[str, float]]) -> None:
    print("\nFR_SARL PPO action-distribution summary")
    print("---------------------------------------")

    for split in SOURCE_SPLITS:
        print(f"\nSplit: {split}")

        for category in DISPLAY_CATEGORIES:
            print(
                f"  {category:35s}: "
                f"{distribution[split][category]:6.2f}%"
            )