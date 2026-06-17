from typing import cast

import numpy as np
from gymnasium.spaces import Discrete

from envs import FDParamControlEnvConfig, create_env
from graph_data import DatasetBuildConfig, build_dataset


def main() -> None:
    """
    Step 11 test.

    This verifies that the final Gymnasium environment connects:

        graph dataset
        algorithm
        action space
        state builder
        reward function
        metrics
        layout context

    into one working RL environment.
    """

    build_dataset(
        config=DatasetBuildConfig(
            output_graph_dir="data/processed/test_env_graphs",
            metadata_path="data/metadata/test_env_dataset_metadata.csv",
            base_seed=2026,
            overwrite=True,
        )
    )

    config = FDParamControlEnvConfig(
        metadata_path="data/metadata/test_env_dataset_metadata.csv",
        split="train",
        layout_scale=1.0,
        max_macro_steps=5,
        iterations_per_step=20,
        seed=42,
        state_name="full",
        action_space_name="pure_fr_multiscale",
        reward_name="aesthetic_delta",
        enable_early_stopping=False,
    )

    env = create_env(
        config=config,
        algorithm_name="fruchterman_reingold",
    )

    if not isinstance(env.action_space, Discrete):
        raise TypeError("Expected env.action_space to be gymnasium.spaces.Discrete.")

    discrete_action_space = cast(Discrete, env.action_space)

    observation, info = env.reset(seed=42)

    assert observation.shape == env.observation_space.shape
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= 0.0)
    assert np.all(observation <= 1.0)

    print("Environment reset passed.")
    print(f"Observation shape: {observation.shape}")
    print(f"Action space size: {discrete_action_space.n}")
    print(f"Initial graph_id: {info['graph_id']}")
    print(f"Initial family: {info['family']}")
    print(f"Initial layout score: {info['layout_score']:.6f}")

    terminated = False
    truncated = False
    step_count = 0

    while not terminated and not truncated:
        action = int(discrete_action_space.sample())

        next_observation, reward, terminated, truncated, step_info = env.step(
            action
        )

        step_count += 1

        assert next_observation.shape == env.observation_space.shape
        assert np.all(np.isfinite(next_observation))
        assert np.all(next_observation >= 0.0)
        assert np.all(next_observation <= 1.0)
        assert np.isfinite(reward)

        print(
            f"\nStep {step_count}"
            f"\n  action_id: {action}"
            f"\n  action_name: {step_info.get('action_name', 'unknown')}"
            f"\n  reward: {reward:.6f}"
            f"\n  layout_score: {step_info['layout_score']:.6f}"
            f"\n  delta_layout_score: {step_info['delta_layout_score']:.6f}"
            f"\n  terminated: {terminated}"
            f"\n  truncated: {truncated}"
        )

        if step_count > config.max_macro_steps + 1:
            raise RuntimeError("Environment did not terminate as expected.")

    assert terminated
    assert step_count == config.max_macro_steps

    env.close()

    print("\nStep 11 environment test passed.")


if __name__ == "__main__":
    main()