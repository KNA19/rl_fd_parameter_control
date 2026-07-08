from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from agents import train_ppo, training_result_to_dict
from experiments.fr_experiment_settings import FR_EXPERIMENT_SETTING
from graph_data import DatasetBuildConfig, build_dataset


def ensure_dataset_exists() -> None:
    """
    Build the default dataset if metadata does not already exist.
    """
    metadata_path = Path(FR_EXPERIMENT_SETTING.metadata_path)

    if metadata_path.exists():
        print(f"Dataset metadata found: {metadata_path}")
        return

    print("Dataset metadata not found. Building default dataset first...")

    build_dataset(
        config=DatasetBuildConfig(
            output_graph_dir="data/processed/graphs",
            metadata_path=str(metadata_path),
            base_seed=2026,
            overwrite=True,
        )
    )


def train_one_seed(
    seed: int,
    check_environment: bool,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Train one PPO model for one random seed.
    """
    model_path = Path(FR_EXPERIMENT_SETTING.model_path_for_seed(seed))

    if model_path.exists() and not force_retrain:
        print("\nSkipping existing model.")
        print(f"Seed: {seed}")
        print(f"Model path: {model_path}")

        return {
            "seed": seed,
            "status": "skipped_existing",
            "model_path": str(model_path),
            "total_timesteps": FR_EXPERIMENT_SETTING.total_timesteps,
            "observation_dim": "",
            "num_actions": "",
            "training_split": "train",
            "training_time_seconds": 0.0,
        }

    model_path.parent.mkdir(parents=True, exist_ok=True)

    training_config = FR_EXPERIMENT_SETTING.make_training_config(seed)

    # Check the environment only for the first seed to save time.
    training_config = replace(
        training_config,
        check_environment=check_environment,
        verbose=1,
    )

    print("\n" + "=" * 80)
    print(f"Training FR-only PPO model for seed {seed}")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Total timesteps: {training_config.total_timesteps}")
    print(f"Max macro-steps: {training_config.max_macro_steps}")
    print(f"Iterations per step: {training_config.iterations_per_step}")
    print(
        "Total FD iterations per episode: "
        f"{training_config.max_macro_steps * training_config.iterations_per_step}"
    )

    start_time = time.time()

    _model, result = train_ppo(training_config)

    elapsed = time.time() - start_time

    row = training_result_to_dict(result)
    row["seed"] = seed
    row["status"] = "trained"
    row["training_time_seconds"] = elapsed

    print(f"\nFinished seed {seed} in {elapsed:.2f} seconds.")

    return row


def save_training_summary(
    rows: List[Dict[str, Any]],
) -> None:
    """
    Save one-row-per-seed training summary.
    """
    output_path = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "training"
        / "fr_multiseed_training_summary.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = [
        "seed",
        "status",
        "model_path",
        "total_timesteps",
        "observation_dim",
        "num_actions",
        "training_split",
        "training_time_seconds",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "seed": row.get("seed", ""),
                    "status": row.get("status", ""),
                    "model_path": row.get("model_path", ""),
                    "total_timesteps": row.get("total_timesteps", ""),
                    "observation_dim": row.get("observation_dim", ""),
                    "num_actions": row.get("num_actions", ""),
                    "training_split": row.get("training_split", ""),
                    "training_time_seconds": row.get(
                        "training_time_seconds",
                        "",
                    ),
                }
            )

    print(f"\nSaved training summary to: {output_path}")


def main() -> None:
    """
    Step 3: Multi-seed FR-only PPO training.

    Trains one PPO model for each seed defined in:

        experiments/fr_experiment_settings.py
    """
    print("FR-only SARL multi-seed training")
    print("--------------------------------")
    print(FR_EXPERIMENT_SETTING.describe())

    ensure_dataset_exists()

    rows: List[Dict[str, Any]] = []

    for index, seed in enumerate(FR_EXPERIMENT_SETTING.training_seeds):
        row = train_one_seed(
            seed=seed,
            check_environment=(index == 0),
            force_retrain=False,
        )

        rows.append(row)

    save_training_summary(rows)

    print("\nStep 3 multi-seed FR PPO training completed.")
    print("Trained/skipped seeds:")

    for row in rows:
        print(
            f"  seed={row['seed']} | "
            f"status={row['status']} | "
            f"model={row['model_path']}"
        )


if __name__ == "__main__":
    main()