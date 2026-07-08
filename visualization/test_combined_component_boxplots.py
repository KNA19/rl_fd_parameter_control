from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Discrete
from matplotlib.axes import Axes
from stable_baselines3 import PPO

from agents.evaluator import get_action_id_by_name
from envs import FDParamControlEnvConfig, ForceDirectedParameterControlEnv, create_env
from experiments.fr_experiment_settings import FR_EXPERIMENT_SETTING


@dataclass(frozen=True)
class ComponentImprovementRecord:
    split: str
    policy: str
    seed: int
    graph_id: str
    component: str
    improvement: float


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

COMPONENTS = (
    "crossing",
    "angular_resolution",
    "edge_length",
    "node_separation",
)

COMPONENT_DISPLAY_NAMES = {
    "crossing": "Crossing score improvement",
    "angular_resolution": "Angular-resolution score improvement",
    "edge_length": "Edge-length score improvement",
    "node_separation": "Node-separation score improvement",
}

COMPONENT_SCORE_KEYS = {
    "crossing": (
        "crossing_score",
        "crossing_quality_score",
        "crossing_component_score",
    ),
    "angular_resolution": (
        "angular_resolution_score",
        "angular_score",
        "angular_component_score",
    ),
    "edge_length": (
        "edge_length_score",
        "edge_length_quality_score",
        "edge_length_component_score",
    ),
    "node_separation": (
        "node_separation_score",
        "node_separation_quality_score",
        "node_separation_component_score",
    ),
}


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


def run_policy_episode(
    env: ForceDirectedParameterControlEnv,
    policy_name: str,
    display_name: str,
    seed: int,
    graph_index: int,
    model: Optional[PPO] = None,
    fixed_action_name: Optional[str] = None,
) -> List[ComponentImprovementRecord]:
    """
    Run one graph episode and return component-wise score improvements.
    """
    if not isinstance(env.action_space, Discrete):
        raise TypeError("Expected env.action_space to be gymnasium.spaces.Discrete.")

    discrete_action_space = cast(Discrete, env.action_space)
    discrete_action_space.seed(seed + graph_index + 999)

    observation, info = env.reset(
        seed=seed + graph_index,
        options={"graph_index": graph_index},
    )

    if env.context is None:
        raise RuntimeError("Environment context was not initialized.")

    initial_scores = dict(env.context.scores)
    graph_id = str(info.get("graph_id", "unknown"))

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action_id = select_action(
            env=env,
            policy_name=policy_name,
            observation=observation,
            discrete_action_space=discrete_action_space,
            model=model,
            fixed_action_name=fixed_action_name,
        )

        observation, _reward, terminated, truncated, _step_info = env.step(action_id)

    if env.context is None:
        raise RuntimeError("Environment context became unavailable.")

    final_scores = dict(env.context.scores)

    records: List[ComponentImprovementRecord] = []

    for component in COMPONENTS:
        initial_value = get_component_score(
            scores=initial_scores,
            component=component,
        )

        final_value = get_component_score(
            scores=final_scores,
            component=component,
        )

        records.append(
            ComponentImprovementRecord(
                split=str(info.get("split", "unknown")),
                policy=display_name,
                seed=seed,
                graph_id=graph_id,
                component=component,
                improvement=final_value - initial_value,
            )
        )

    return records


def select_action(
    env: ForceDirectedParameterControlEnv,
    policy_name: str,
    observation: np.ndarray,
    discrete_action_space: Discrete,
    model: Optional[PPO],
    fixed_action_name: Optional[str],
) -> int:
    """
    Select one action for the current policy.
    """
    if policy_name == "no_change":
        return get_action_id_by_name(
            env=env,
            action_name="no_change",
        )

    if policy_name == "random":
        return int(discrete_action_space.sample())

    if policy_name == "fixed":
        if fixed_action_name is None:
            raise ValueError("fixed_action_name is required for fixed policy.")

        return get_action_id_by_name(
            env=env,
            action_name=fixed_action_name,
        )

    if policy_name == "ppo":
        if model is None:
            raise ValueError("model is required for PPO policy.")

        action, _state = model.predict(
            observation,
            deterministic=True,
        )

        action_id = int(np.asarray(action).reshape(-1)[0])

        if action_id < 0 or action_id >= int(discrete_action_space.n):
            raise ValueError(f"Invalid PPO action id: {action_id}")

        return action_id

    raise ValueError(
        f"Unknown policy_name={policy_name}. "
        "Available: no_change, random, fixed, ppo."
    )


def get_component_score(
    scores: Dict[str, float],
    component: str,
) -> float:
    """
    Extract a component score from LayoutScoreCalculator output.

    If your project uses slightly different score-key names, add the key
    to COMPONENT_SCORE_KEYS above.
    """
    candidate_keys = COMPONENT_SCORE_KEYS[component]

    for key in candidate_keys:
        if key in scores:
            return float(scores[key])

    available_keys = ", ".join(sorted(scores.keys()))

    raise KeyError(
        f"Could not find score key for component '{component}'.\n"
        f"Tried: {candidate_keys}\n"
        f"Available score keys: {available_keys}"
    )


def collect_combined_component_records() -> List[ComponentImprovementRecord]:
    """
    Collect component-wise improvement records from all test splits together.

    This reruns evaluation episodes so that we can access component scores,
    not only the aggregate layout score saved in previous CSVs.
    """
    best_fixed_by_split = get_best_fixed_action_by_split()

    all_records: List[ComponentImprovementRecord] = []

    for split in TEST_SPLITS:
        best_fixed_action = best_fixed_by_split[split]

        for seed in FR_EXPERIMENT_SETTING.training_seeds:
            model_path = Path(FR_EXPERIMENT_SETTING.model_path_for_seed(seed))

            if not model_path.exists():
                raise FileNotFoundError(
                    f"PPO model not found: {model_path}\n"
                    "Run Step 3 first:\n"
                    "python -m experiments.run_fr_multiseed_training"
                )

            env_config = make_env_config(
                seed=seed,
                split=split,
            )

            env = create_env(
                config=env_config,
                algorithm_name=FR_EXPERIMENT_SETTING.algorithm_name,
            )

            ppo_model = PPO.load(
                path=str(model_path),
                env=env,
            )

            num_available_graphs = len(env.metadata_rows)

            for episode_index in range(FR_EXPERIMENT_SETTING.evaluation_episodes):
                graph_index = episode_index % num_available_graphs

                all_records.extend(
                    run_policy_episode(
                        env=env,
                        policy_name="no_change",
                        display_name="Default FR",
                        seed=seed,
                        graph_index=graph_index,
                    )
                )

                all_records.extend(
                    run_policy_episode(
                        env=env,
                        policy_name="random",
                        display_name="Random",
                        seed=seed,
                        graph_index=graph_index,
                    )
                )

                all_records.extend(
                    run_policy_episode(
                        env=env,
                        policy_name="fixed",
                        display_name="Best fixed",
                        seed=seed,
                        graph_index=graph_index,
                        fixed_action_name=best_fixed_action,
                    )
                )

                all_records.extend(
                    run_policy_episode(
                        env=env,
                        policy_name="ppo",
                        display_name="PPO",
                        seed=seed,
                        graph_index=graph_index,
                        model=ppo_model,
                    )
                )

            env.close()

    return all_records


def save_component_data(
    records: Sequence[ComponentImprovementRecord],
    output_csv_path: Path,
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "split": record.split,
            "policy": record.policy,
            "seed": record.seed,
            "graph_id": record.graph_id,
            "component": record.component,
            "component_improvement": record.improvement,
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

    print(f"Saved Figure 3 component data to: {output_csv_path}")


def plot_component_boxplots(
    records: Sequence[ComponentImprovementRecord],
    output_path: Path,
) -> None:
    """
    Generate Figure 3:

    One combined test-class figure with component-wise improvement box plots.

    Layout:
        2 x 2 panels

    Components:
        crossing
        angular resolution
        edge length
        node separation
    """
    if not records:
        raise ValueError("No records provided for plotting.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.2),
        squeeze=False,
    )

    axis_list = [
        axes[0][0],
        axes[0][1],
        axes[1][0],
        axes[1][1],
    ]

    for axis, component in zip(axis_list, COMPONENTS):
        plot_one_component(
            axis=axis,
            records=records,
            component=component,
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.98,
        bottom=0.10,
        hspace=0.42,
        wspace=0.28,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 3 component box plots to: {output_path}")


def plot_one_component(
    axis: Axes,
    records: Sequence[ComponentImprovementRecord],
    component: str,
) -> None:
    data: List[List[float]] = []

    for policy in POLICIES:
        values = [
            record.improvement
            for record in records
            if record.component == component and record.policy == policy
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
    axis.set_ylabel(
        COMPONENT_DISPLAY_NAMES[component],
        fontsize=9,
    )

    axis.grid(axis="y", alpha=0.30)
    axis.tick_params(axis="x", labelrotation=20, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)


def generate_figure3_combined_test_components() -> None:
    records = collect_combined_component_records()

    output_dir = (
        Path(FR_EXPERIMENT_SETTING.output_root)
        / "figures"
        / "figure_3_combined_test_components"
    )

    save_component_data(
        records=records,
        output_csv_path=output_dir / "figure_3_combined_test_component_data.csv",
    )

    plot_component_boxplots(
        records=records,
        output_path=output_dir / "figure_3_combined_test_component_boxplots.png",
    )

    print_summary(records)


def print_summary(records: Sequence[ComponentImprovementRecord]) -> None:
    print("\nFigure 3 component-wise improvement summary")
    print("-------------------------------------------")

    for component in COMPONENTS:
        print(f"\nComponent: {component}")

        for policy in POLICIES:
            values = [
                record.improvement
                for record in records
                if record.component == component and record.policy == policy
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