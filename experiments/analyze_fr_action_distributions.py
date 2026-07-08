from __future__ import annotations

import ast
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from experiments.fr_experiment_settings import FR_EXPERIMENT_SETTING


ActionCountDict = Dict[str, int]
RowDict = Dict[str, Any]


def parse_action_counts(raw_value: str) -> ActionCountDict:
    """
    Parse the action_counts column from policy comparison CSV files.

    The CSV stores action_counts as a string representation of a dictionary,
    for example:

        "{'large_increase_k': 120, 'reset_cooling_rate_to_default': 30}"
    """
    if raw_value is None:
        return {}

    text = str(raw_value).strip()

    if not text:
        return {}

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}

    if not isinstance(parsed, dict):
        return {}

    output: ActionCountDict = {}

    for key, value in parsed.items():
        try:
            output[str(key)] = int(value)
        except (TypeError, ValueError):
            output[str(key)] = 0

    return output


def read_ppo_action_counts(
    seed: int,
    split: str,
) -> ActionCountDict:
    """
    Read PPO action counts from one seed/split comparison CSV.
    """
    csv_path = Path(
        FR_EXPERIMENT_SETTING.comparison_csv_path(
            seed=seed,
            split=split,
        )
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing comparison CSV: {csv_path}\n"
            "Run Step 4 first:\n"
            "python -m experiments.run_fr_multiseed_evaluation"
        )

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if str(row.get("policy", "")).strip() == "ppo":
                return parse_action_counts(row.get("action_counts", ""))

    raise ValueError(f"No PPO row found in: {csv_path}")


def compute_entropy(
    counts: ActionCountDict,
) -> float:
    """
    Compute normalized action entropy.

    0 means one action dominates completely.
    1 means actions are used uniformly.
    """
    total = sum(counts.values())

    if total <= 0:
        return 0.0

    positive_counts = [count for count in counts.values() if count > 0]

    if len(positive_counts) <= 1:
        return 0.0

    entropy = 0.0

    for count in positive_counts:
        probability = count / total
        entropy -= probability * math.log(probability)

    max_entropy = math.log(len(counts))

    if max_entropy <= 0.0:
        return 0.0

    return entropy / max_entropy


def effective_number_of_actions(
    counts: ActionCountDict,
) -> float:
    """
    Compute effective number of actions using exp(entropy).

    This gives an interpretable diversity estimate.
    """
    total = sum(counts.values())

    if total <= 0:
        return 0.0

    entropy = 0.0

    for count in counts.values():
        if count <= 0:
            continue

        probability = count / total
        entropy -= probability * math.log(probability)

    return math.exp(entropy)


def max_action_share(
    counts: ActionCountDict,
) -> float:
    """
    Return fraction of decisions taken by the most frequent action.
    """
    total = sum(counts.values())

    if total <= 0:
        return 0.0

    return max(counts.values()) / total


def complete_action_counts(
    counts: ActionCountDict,
) -> ActionCountDict:
    """
    Ensure every valid FR action appears, even if PPO used it zero times.
    """
    completed: ActionCountDict = {}

    for action_name in FR_EXPERIMENT_SETTING.fixed_action_names:
        completed[action_name] = int(counts.get(action_name, 0))

    # Keep any unexpected action too, for safety.
    for action_name, count in counts.items():
        if action_name not in completed:
            completed[action_name] = int(count)

    return completed


def save_csv(
    rows: List[RowDict],
    output_path: Path,
) -> None:
    """
    Save rows to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


def build_seed_split_rows() -> Tuple[List[RowDict], List[RowDict]]:
    """
    Build detailed action-count rows and diversity rows per seed/split.
    """
    action_rows: List[RowDict] = []
    diversity_rows: List[RowDict] = []

    for seed in FR_EXPERIMENT_SETTING.training_seeds:
        for split in FR_EXPERIMENT_SETTING.evaluation_splits:
            counts = complete_action_counts(
                read_ppo_action_counts(
                    seed=seed,
                    split=split,
                )
            )

            total_actions = sum(counts.values())
            entropy = compute_entropy(counts)
            effective_actions = effective_number_of_actions(counts)
            max_share = max_action_share(counts)

            ranked_actions = sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_action = ranked_actions[0][0] if ranked_actions else ""
            top_count = ranked_actions[0][1] if ranked_actions else 0

            diversity_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "total_actions": total_actions,
                    "unique_actions_used": sum(
                        1 for count in counts.values() if count > 0
                    ),
                    "top_action": top_action,
                    "top_action_count": top_count,
                    "top_action_share": max_share,
                    "normalized_entropy": entropy,
                    "effective_number_of_actions": effective_actions,
                }
            )

            for rank, (action_name, count) in enumerate(ranked_actions, start=1):
                percentage = (
                    100.0 * count / total_actions
                    if total_actions > 0
                    else 0.0
                )

                action_rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "rank": rank,
                        "action_name": action_name,
                        "count": count,
                        "percentage": percentage,
                    }
                )

    return action_rows, diversity_rows


def build_split_aggregate_rows() -> Tuple[List[RowDict], List[RowDict], List[RowDict]]:
    """
    Aggregate PPO action counts across seeds for each split.
    """
    aggregate_action_rows: List[RowDict] = []
    aggregate_diversity_rows: List[RowDict] = []
    top_action_rows: List[RowDict] = []

    for split in FR_EXPERIMENT_SETTING.evaluation_splits:
        split_counter: Counter[str] = Counter()

        for seed in FR_EXPERIMENT_SETTING.training_seeds:
            counts = complete_action_counts(
                read_ppo_action_counts(
                    seed=seed,
                    split=split,
                )
            )

            split_counter.update(counts)

        completed_counts = complete_action_counts(dict(split_counter))
        total_actions = sum(completed_counts.values())

        entropy = compute_entropy(completed_counts)
        effective_actions = effective_number_of_actions(completed_counts)
        max_share = max_action_share(completed_counts)

        ranked_actions = sorted(
            completed_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_action = ranked_actions[0][0] if ranked_actions else ""
        top_count = ranked_actions[0][1] if ranked_actions else 0

        aggregate_diversity_rows.append(
            {
                "split": split,
                "num_seeds": len(FR_EXPERIMENT_SETTING.training_seeds),
                "total_actions": total_actions,
                "unique_actions_used": sum(
                    1 for count in completed_counts.values() if count > 0
                ),
                "top_action": top_action,
                "top_action_count": top_count,
                "top_action_share": max_share,
                "normalized_entropy": entropy,
                "effective_number_of_actions": effective_actions,
            }
        )

        for rank, (action_name, count) in enumerate(ranked_actions, start=1):
            percentage = (
                100.0 * count / total_actions
                if total_actions > 0
                else 0.0
            )

            row = {
                "split": split,
                "rank": rank,
                "action_name": action_name,
                "count": count,
                "percentage": percentage,
            }

            aggregate_action_rows.append(row)

            if rank <= 5:
                top_action_rows.append(row)

    return aggregate_action_rows, aggregate_diversity_rows, top_action_rows


def print_action_distribution_summary(
    aggregate_diversity_rows: List[RowDict],
    top_action_rows: List[RowDict],
) -> None:
    """
    Print compact terminal summary.
    """
    print("\nPPO action-distribution summary")
    print("-------------------------------")

    for row in aggregate_diversity_rows:
        split = row["split"]

        print(f"\nSplit: {split}")
        print(f"  Total PPO decisions: {row['total_actions']}")
        print(f"  Unique actions used: {row['unique_actions_used']}")
        print(
            f"  Top action: {row['top_action']} "
            f"({100.0 * row['top_action_share']:.2f}%)"
        )
        print(f"  Normalized entropy: {row['normalized_entropy']:.4f}")
        print(
            "  Effective number of actions: "
            f"{row['effective_number_of_actions']:.2f}"
        )

        print("  Top 5 actions:")

        split_top_rows = [
            item for item in top_action_rows
            if item["split"] == split
        ]

        for item in split_top_rows:
            print(
                f"    {item['rank']:02d}. "
                f"{item['action_name']:35s} "
                f"{item['count']:5d} "
                f"({item['percentage']:.2f}%)"
            )


def main() -> None:
    """
    Step 6: Analyze PPO action distributions.

    This script reads PPO action_counts from the Step 4 comparison CSVs and
    creates action-distribution summaries across seeds and splits.
    """
    print("FR-only PPO action distribution analysis")
    print("----------------------------------------")

    action_rows, diversity_rows = build_seed_split_rows()

    (
        aggregate_action_rows,
        aggregate_diversity_rows,
        top_action_rows,
    ) = build_split_aggregate_rows()

    output_dir = Path(FR_EXPERIMENT_SETTING.output_root) / "aggregate"

    save_csv(
        rows=action_rows,
        output_path=output_dir
        / "fr_sarl_v1_ppo_action_distribution_by_seed_split.csv",
    )

    save_csv(
        rows=diversity_rows,
        output_path=output_dir
        / "fr_sarl_v1_ppo_action_diversity_by_seed_split.csv",
    )

    save_csv(
        rows=aggregate_action_rows,
        output_path=output_dir
        / "fr_sarl_v1_ppo_action_distribution_by_split.csv",
    )

    save_csv(
        rows=aggregate_diversity_rows,
        output_path=output_dir
        / "fr_sarl_v1_ppo_action_diversity_by_split.csv",
    )

    save_csv(
        rows=top_action_rows,
        output_path=output_dir
        / "fr_sarl_v1_ppo_top_actions_by_split.csv",
    )

    print_action_distribution_summary(
        aggregate_diversity_rows=aggregate_diversity_rows,
        top_action_rows=top_action_rows,
    )

    print("\nStep 6 action-distribution analysis completed.")


if __name__ == "__main__":
    main()