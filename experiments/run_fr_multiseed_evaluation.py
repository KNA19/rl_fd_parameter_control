from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from agents import (
    evaluate_fixed_action_policy,
    evaluate_ppo,
    evaluate_random_policy,
    save_comparison_csv,
)
from experiments.fr_experiment_settings import FR_EXPERIMENT_SETTING


SummaryRow = Dict[str, Any]


def require_trained_models() -> None:
    """
    Verify that all PPO models exist before evaluation begins.
    """
    missing_models: List[str] = []

    for seed in FR_EXPERIMENT_SETTING.training_seeds:
        model_path = Path(FR_EXPERIMENT_SETTING.model_path_for_seed(seed))

        if not model_path.exists():
            missing_models.append(str(model_path))

    if missing_models:
        message = "\n".join(missing_models)
        raise FileNotFoundError(
            "Some trained PPO models are missing:\n"
            f"{message}\n\n"
            "Run Step 3 first:\n"
            "python -m experiments.run_fr_multiseed_training"
        )


def evaluate_one_seed_split(
    seed: int,
    split: str,
) -> List[SummaryRow]:
    """
    Evaluate all policies for one seed and one split.

    Policies:
        - every fixed-action policy
        - random policy
        - PPO policy
    """
    print("\n" + "=" * 90)
    print(f"Evaluating seed={seed}, split={split}")
    print("=" * 90)

    summaries: List[SummaryRow] = []

    # 1. Fixed-action sweep.
    for action_name in FR_EXPERIMENT_SETTING.fixed_action_names:
        policy_name = f"fixed::{action_name}"

        config = FR_EXPERIMENT_SETTING.make_evaluation_config(
            seed=seed,
            split=split,
            policy_name=policy_name,
        )

        summary = evaluate_fixed_action_policy(
            config=config,
            action_name=action_name,
        )

        summaries.append(summary)

    # 2. Random baseline.
    random_config = FR_EXPERIMENT_SETTING.make_evaluation_config(
        seed=seed,
        split=split,
        policy_name="random",
    )

    random_summary = evaluate_random_policy(
        config=random_config,
    )

    summaries.append(random_summary)

    # 3. PPO.
    ppo_config = FR_EXPERIMENT_SETTING.make_evaluation_config(
        seed=seed,
        split=split,
        policy_name="ppo",
    )

    ppo_summary = evaluate_ppo(
        config=ppo_config,
    )

    summaries.append(ppo_summary)

    # Save one-row-per-policy comparison for this seed/split.
    save_comparison_csv(
        summaries=summaries,
        output_csv_path=FR_EXPERIMENT_SETTING.comparison_csv_path(
            seed=seed,
            split=split,
        ),
    )

    # Save fixed-action-only sweep for this seed/split.
    fixed_summaries = [
        summary
        for summary in summaries
        if str(summary.get("policy", "")).startswith("fixed::")
    ]

    save_comparison_csv(
        summaries=fixed_summaries,
        output_csv_path=FR_EXPERIMENT_SETTING.fixed_action_sweep_csv_path(
            seed=seed,
            split=split,
        ),
    )

    print_policy_ranking(
        summaries=summaries,
        seed=seed,
        split=split,
    )

    return summaries


def print_policy_ranking(
    summaries: List[SummaryRow],
    seed: int,
    split: str,
) -> None:
    """
    Print ranking by mean layout-score improvement.
    """
    ranked = sorted(
        summaries,
        key=lambda item: float(item["mean_layout_score_improvement"]),
        reverse=True,
    )

    print("\nPolicy ranking")
    print("--------------")
    print(f"Seed: {seed}")
    print(f"Split: {split}")

    for rank, summary in enumerate(ranked[:10], start=1):
        print(
            f"{rank:02d}. "
            f"{summary['policy']:35s} | "
            f"improvement={summary['mean_layout_score_improvement']:.6f} | "
            f"final={summary['mean_final_layout_score']:.6f} | "
            f"reward={summary['mean_total_reward']:.6f}"
        )


def aggregate_across_seeds(
    all_summaries: List[SummaryRow],
) -> List[SummaryRow]:
    """
    Aggregate policy results across seeds for each split.

    Output rows are grouped by:
        split, policy
    """
    grouped: Dict[Tuple[str, str], List[SummaryRow]] = defaultdict(list)

    for summary in all_summaries:
        split = str(summary["split"])
        policy = str(summary["policy"])
        grouped[(split, policy)].append(summary)

    aggregate_rows: List[SummaryRow] = []

    for (split, policy), rows in sorted(grouped.items()):
        improvements = _values(rows, "mean_layout_score_improvement")
        final_scores = _values(rows, "mean_final_layout_score")
        initial_scores = _values(rows, "mean_initial_layout_score")
        total_rewards = _values(rows, "mean_total_reward")
        episode_lengths = _values(rows, "mean_episode_length")

        aggregate_rows.append(
            {
                "split": split,
                "policy": policy,
                "num_seeds": len(rows),
                "mean_initial_layout_score": float(np.mean(initial_scores)),
                "std_initial_layout_score": float(np.std(initial_scores, ddof=0)),
                "mean_final_layout_score": float(np.mean(final_scores)),
                "std_final_layout_score": float(np.std(final_scores, ddof=0)),
                "mean_layout_score_improvement": float(np.mean(improvements)),
                "std_layout_score_improvement": float(np.std(improvements, ddof=0)),
                "mean_total_reward": float(np.mean(total_rewards)),
                "std_total_reward": float(np.std(total_rewards, ddof=0)),
                "mean_episode_length": float(np.mean(episode_lengths)),
                "std_episode_length": float(np.std(episode_lengths, ddof=0)),
            }
        )

    return aggregate_rows


def save_aggregate_summary(
    rows: List[SummaryRow],
    output_csv_path: str,
) -> None:
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = [
        "split",
        "policy",
        "num_seeds",
        "mean_initial_layout_score",
        "std_initial_layout_score",
        "mean_final_layout_score",
        "std_final_layout_score",
        "mean_layout_score_improvement",
        "std_layout_score_improvement",
        "mean_total_reward",
        "std_total_reward",
        "mean_episode_length",
        "std_episode_length",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved aggregate summary to: {output_path}")


def save_best_policy_summary(
    aggregate_rows: List[SummaryRow],
) -> None:
    """
    Save best policy per split based on mean layout-score improvement.
    """
    output_path = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "aggregate"
        / "fr_sarl_v1_best_policy_by_split.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[SummaryRow]] = defaultdict(list)

    for row in aggregate_rows:
        grouped[str(row["split"])].append(row)

    best_rows: List[SummaryRow] = []

    for split, rows in sorted(grouped.items()):
        best = max(
            rows,
            key=lambda item: float(item["mean_layout_score_improvement"]),
        )

        ppo_rows = [
            row for row in rows
            if str(row["policy"]) == "ppo"
        ]

        ppo_improvement = (
            float(ppo_rows[0]["mean_layout_score_improvement"])
            if ppo_rows
            else float("nan")
        )

        best_rows.append(
            {
                "split": split,
                "best_policy": best["policy"],
                "best_mean_improvement": best["mean_layout_score_improvement"],
                "best_std_improvement": best["std_layout_score_improvement"],
                "ppo_mean_improvement": ppo_improvement,
                "ppo_minus_best": ppo_improvement
                - float(best["mean_layout_score_improvement"]),
            }
        )

    fieldnames = [
        "split",
        "best_policy",
        "best_mean_improvement",
        "best_std_improvement",
        "ppo_mean_improvement",
        "ppo_minus_best",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(best_rows)

    print(f"Saved best-policy summary to: {output_path}")


def print_aggregate_highlights(
    aggregate_rows: List[SummaryRow],
) -> None:
    """
    Print compact highlights for PPO, no_change, random, and best fixed action.
    """
    print("\nAggregate highlights")
    print("--------------------")

    splits = sorted(set(str(row["split"]) for row in aggregate_rows))

    for split in splits:
        rows = [
            row for row in aggregate_rows
            if str(row["split"]) == split
        ]

        ppo = _find_policy(rows, "ppo")
        random = _find_policy(rows, "random")
        no_change = _find_policy(rows, "fixed::no_change")

        fixed_rows = [
            row for row in rows
            if str(row["policy"]).startswith("fixed::")
        ]

        best_fixed = max(
            fixed_rows,
            key=lambda item: float(item["mean_layout_score_improvement"]),
        )

        print(f"\nSplit: {split}")

        for label, row in [
            ("PPO", ppo),
            ("No-change/default FR", no_change),
            ("Random", random),
            ("Best fixed action", best_fixed),
        ]:
            if row is None:
                print(f"  {label:24s}: missing")
                continue

            print(
                f"  {label:24s}: "
                f"{row['policy']:35s} | "
                f"improvement="
                f"{row['mean_layout_score_improvement']:.6f} "
                f"± {row['std_layout_score_improvement']:.6f}"
            )


def _find_policy(
    rows: Iterable[SummaryRow],
    policy_name: str,
) -> SummaryRow | None:
    for row in rows:
        if str(row["policy"]) == policy_name:
            return row

    return None


def _values(
    rows: List[SummaryRow],
    key: str,
) -> List[float]:
    return [float(row[key]) for row in rows]


def main() -> None:
    """
    Step 4: Evaluate all trained FR PPO models on all splits.

    For each seed and split, evaluates:
        - all fixed-action policies
        - random policy
        - PPO policy

    Then aggregates results across seeds.
    """
    print("FR-only SARL multi-seed evaluation")
    print("----------------------------------")
    print(FR_EXPERIMENT_SETTING.describe())

    require_trained_models()

    all_summaries: List[SummaryRow] = []

    for seed in FR_EXPERIMENT_SETTING.training_seeds:
        for split in FR_EXPERIMENT_SETTING.evaluation_splits:
            summaries = evaluate_one_seed_split(
                seed=seed,
                split=split,
            )

            all_summaries.extend(summaries)

    aggregate_rows = aggregate_across_seeds(all_summaries)

    save_aggregate_summary(
        rows=aggregate_rows,
        output_csv_path=FR_EXPERIMENT_SETTING.aggregate_csv_path(),
    )

    save_best_policy_summary(aggregate_rows)

    print_aggregate_highlights(aggregate_rows)

    print("\nStep 4 FR-only multi-seed evaluation completed.")


if __name__ == "__main__":
    main()