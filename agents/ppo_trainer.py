from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs import FDParamControlEnvConfig, ForceDirectedParameterControlEnv, create_env


@dataclass(frozen=True)
class PPOTrainingConfig:
    """
    Configuration for PPO training.

    Debug setting:
        total_timesteps = 1000 or 2000
        max_macro_steps = 5
        iterations_per_step = 20

    Final experiment setting:
        total_timesteps = 100000+
        max_macro_steps = 50
        iterations_per_step = 50
    """

    metadata_path: str = "data/metadata/dataset_metadata.csv"
    split: str = "train"

    layout_scale: float = 1.0
    max_macro_steps: int = 5
    iterations_per_step: int = 20

    seed: int = 2026

    algorithm_name: str = "fruchterman_reingold"
    state_name: str = "full"
    action_space_name: str = "pure_fr_multiscale"
    reward_name: str = "aesthetic_delta"

    total_timesteps: int = 2000

    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.01
    vf_coef: float = 0.50

    model_output_path: str = "outputs/models/ppo/fd_param_control_ppo.zip"
    tensorboard_log_path: Optional[str] = None

    check_environment: bool = True
    verbose: int = 1


@dataclass(frozen=True)
class PPOTrainingResult:
    """
    Output summary after PPO training.
    """

    model_path: str
    total_timesteps: int
    observation_dim: int
    num_actions: int
    training_split: str


def make_training_env(
    config: PPOTrainingConfig,
) -> ForceDirectedParameterControlEnv:
    """
    Create the training environment from PPOTrainingConfig.
    """
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

    return create_env(
        config=env_config,
        algorithm_name=config.algorithm_name,
    )


def get_environment_dimensions(
    env: ForceDirectedParameterControlEnv,
) -> Tuple[int, int]:
    """
    Safely extract observation dimension and number of actions.

    This avoids Pylance errors because Gymnasium stores spaces using
    general Space types.
    """
    if not isinstance(env.observation_space, Box):
        raise TypeError("Expected env.observation_space to be gymnasium.spaces.Box.")

    if not isinstance(env.action_space, Discrete):
        raise TypeError("Expected env.action_space to be gymnasium.spaces.Discrete.")

    observation_space = cast(Box, env.observation_space)
    action_space = cast(Discrete, env.action_space)

    observation_shape = tuple(observation_space.shape)

    if len(observation_shape) != 1:
        raise ValueError(
            f"Expected one-dimensional observation space, got {observation_shape}."
        )

    observation_dim = int(observation_shape[0])
    num_actions = int(action_space.n)

    return observation_dim, num_actions


def train_ppo(
    config: PPOTrainingConfig,
) -> Tuple[PPO, PPOTrainingResult]:
    """
    Train PPO on the final force-directed parameter-control environment.
    """
    env = make_training_env(config)

    if config.check_environment:
        print("Checking Gymnasium environment with Stable-Baselines3 check_env()...")
        check_env(env, warn=True)
        print("Environment check completed.")

    observation_dim, num_actions = get_environment_dimensions(env)

    model_output_path = Path(config.model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    tensorboard_log = config.tensorboard_log_path

    if tensorboard_log is not None:
        Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    print("\nStarting PPO training...")
    print(f"Training split: {config.split}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Observation dimension: {observation_dim}")
    print(f"Action-space size: {num_actions}")
    print(f"Model output path: {model_output_path}")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        seed=config.seed,
        verbose=config.verbose,
        tensorboard_log=tensorboard_log,
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        progress_bar=False,
    )

    model.save(str(model_output_path))

    result = PPOTrainingResult(
        model_path=str(model_output_path),
        total_timesteps=config.total_timesteps,
        observation_dim=observation_dim,
        num_actions=num_actions,
        training_split=config.split,
    )

    env.close()

    print("\nPPO training completed.")
    print(f"Saved model to: {model_output_path}")

    return model, result


def training_result_to_dict(
    result: PPOTrainingResult,
) -> Dict[str, Any]:
    return {
        "model_path": result.model_path,
        "total_timesteps": result.total_timesteps,
        "observation_dim": result.observation_dim,
        "num_actions": result.num_actions,
        "training_split": result.training_split,
    }