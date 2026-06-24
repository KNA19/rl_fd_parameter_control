from __future__ import annotations

from typing import List, Tuple

from algorithms.base import ParameterSpace
from actions.base_action_space import (
    ActionSpec,
    BaseParameterActionSpace,
    ParameterOperation,
)


class MultiScaleParameterActionSpace(BaseParameterActionSpace):
    """
    Multi-scale discrete parameter-control action space.

    This class supports:

        generic single-parameter actions
        algorithm-specific coupled actions

    For Pure FR, the expected parameters are:

        k
        temperature
        cooling_rate
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        algorithm_name: str = "generic",
    ):
        self.algorithm_name = algorithm_name

        action_specs = build_multiscale_action_specs(
            parameter_space=parameter_space,
            algorithm_name=algorithm_name,
        )

        super().__init__(
            parameter_space=parameter_space,
            action_specs=action_specs,
        )


def build_multiscale_action_specs(
    parameter_space: ParameterSpace,
    algorithm_name: str = "generic",
) -> Tuple[ActionSpec, ...]:
    """
    Build action specifications for a given algorithm.

    Generic part:
        no_change
        small/large increase/decrease for each parameter
        reset each parameter

    Pure FR additions:
        expand_and_explore
        expand_and_stabilize
        compress_and_stabilize
        reheat_layout
        cool_down_layout
        reset_k_to_default
    """
    specs: List[ActionSpec] = []

    specs.append(
        ActionSpec(
            action_id=len(specs),
            name="no_change",
            description="Keep all parameters unchanged.",
            operations=(),
        )
    )

    for parameter_name in parameter_space.parameter_names:
        specs.extend(
            _single_parameter_actions(
                start_id=len(specs),
                parameter_name=parameter_name,
            )
        )

    for parameter_name in parameter_space.parameter_names:
        specs.append(
            ActionSpec(
                action_id=len(specs),
                name=f"reset_{parameter_name}_to_default",
                description=f"Reset {parameter_name} to its default value.",
                operations=(
                    ParameterOperation(
                        parameter_name=parameter_name,
                        operation_type="reset",
                    ),
                ),
            )
        )

    if _is_pure_fr_action_space(
        parameter_space=parameter_space,
        algorithm_name=algorithm_name,
    ):
        specs.extend(
            _pure_fr_coupled_actions(
                start_id=len(specs),
            )
        )

    return tuple(specs)


def _single_parameter_actions(
    start_id: int,
    parameter_name: str,
) -> List[ActionSpec]:
    """
    Create multi-scale increase/decrease actions for one parameter.

    Cooling-rate-like parameters are updated additively because their natural
    range is usually close to [0, 1].

    Other parameters are updated multiplicatively.
    """
    lower_name = parameter_name.lower()

    if "cooling" in lower_name:
        return [
            ActionSpec(
                action_id=start_id,
                name=f"small_increase_{parameter_name}",
                description=f"Slightly increase {parameter_name}.",
                operations=(
                    ParameterOperation(
                        parameter_name=parameter_name,
                        operation_type="add",
                        value=0.01,
                    ),
                ),
            ),
            ActionSpec(
                action_id=start_id + 1,
                name=f"small_decrease_{parameter_name}",
                description=f"Slightly decrease {parameter_name}.",
                operations=(
                    ParameterOperation(
                        parameter_name=parameter_name,
                        operation_type="add",
                        value=-0.01,
                    ),
                ),
            ),
            ActionSpec(
                action_id=start_id + 2,
                name=f"large_increase_{parameter_name}",
                description=f"Strongly increase {parameter_name}.",
                operations=(
                    ParameterOperation(
                        parameter_name=parameter_name,
                        operation_type="add",
                        value=0.03,
                    ),
                ),
            ),
            ActionSpec(
                action_id=start_id + 3,
                name=f"large_decrease_{parameter_name}",
                description=f"Strongly decrease {parameter_name}.",
                operations=(
                    ParameterOperation(
                        parameter_name=parameter_name,
                        operation_type="add",
                        value=-0.03,
                    ),
                ),
            ),
        ]

    return [
        ActionSpec(
            action_id=start_id,
            name=f"small_increase_{parameter_name}",
            description=f"Slightly increase {parameter_name}.",
            operations=(
                ParameterOperation(
                    parameter_name=parameter_name,
                    operation_type="multiply",
                    value=1.05,
                ),
            ),
        ),
        ActionSpec(
            action_id=start_id + 1,
            name=f"small_decrease_{parameter_name}",
            description=f"Slightly decrease {parameter_name}.",
            operations=(
                ParameterOperation(
                    parameter_name=parameter_name,
                    operation_type="multiply",
                    value=0.95,
                ),
            ),
        ),
        ActionSpec(
            action_id=start_id + 2,
            name=f"large_increase_{parameter_name}",
            description=f"Strongly increase {parameter_name}.",
            operations=(
                ParameterOperation(
                    parameter_name=parameter_name,
                    operation_type="multiply",
                    value=1.20,
                ),
            ),
        ),
        ActionSpec(
            action_id=start_id + 3,
            name=f"large_decrease_{parameter_name}",
            description=f"Strongly decrease {parameter_name}.",
            operations=(
                ParameterOperation(
                    parameter_name=parameter_name,
                    operation_type="multiply",
                    value=0.80,
                ),
            ),
        ),
    ]


def _is_pure_fr_action_space(
    parameter_space: ParameterSpace,
    algorithm_name: str,
) -> bool:
    """
    Return True if this appears to be the Pure FR parameter space.
    """
    names = set(parameter_space.parameter_names)

    has_fr_parameters = {
        "k",
        "temperature",
        "cooling_rate",
    }.issubset(names)

    normalized_algorithm_name = algorithm_name.lower().strip()

    return has_fr_parameters and (
        normalized_algorithm_name in {
            "fruchterman_reingold",
            "pure_fr",
            "fr",
            "generic",
        }
    )


def _pure_fr_coupled_actions(
    start_id: int,
) -> List[ActionSpec]:
    """
    Pure FR-specific coupled parameter actions.

    These actions still do not move nodes directly.
    They only modify Pure FR parameters.

    Note:
        reset_k_to_default is not included here because generic reset actions
        are already created for every parameter.
    """
    return [
        ActionSpec(
            action_id=start_id,
            name="expand_and_explore",
            description=(
                "Increase k and temperature to encourage expansion and movement."
            ),
            operations=(
                ParameterOperation(
                    parameter_name="k",
                    operation_type="multiply",
                    value=1.10,
                ),
                ParameterOperation(
                    parameter_name="temperature",
                    operation_type="multiply",
                    value=1.10,
                ),
            ),
        ),
        ActionSpec(
            action_id=start_id + 1,
            name="expand_and_stabilize",
            description=(
                "Increase k but reduce temperature to expand while stabilizing."
            ),
            operations=(
                ParameterOperation(
                    parameter_name="k",
                    operation_type="multiply",
                    value=1.10,
                ),
                ParameterOperation(
                    parameter_name="temperature",
                    operation_type="multiply",
                    value=0.90,
                ),
            ),
        ),
        ActionSpec(
            action_id=start_id + 2,
            name="compress_and_stabilize",
            description=(
                "Decrease k and temperature to reduce over-expansion and stabilize."
            ),
            operations=(
                ParameterOperation(
                    parameter_name="k",
                    operation_type="multiply",
                    value=0.95,
                ),
                ParameterOperation(
                    parameter_name="temperature",
                    operation_type="multiply",
                    value=0.90,
                ),
            ),
        ),
        ActionSpec(
            action_id=start_id + 3,
            name="reheat_layout",
            description=(
                "Increase temperature and slow cooling to escape stagnation."
            ),
            operations=(
                ParameterOperation(
                    parameter_name="temperature",
                    operation_type="multiply",
                    value=1.40,
                ),
                ParameterOperation(
                    parameter_name="cooling_rate",
                    operation_type="add",
                    value=0.01,
                ),
            ),
        ),
        ActionSpec(
            action_id=start_id + 4,
            name="cool_down_layout",
            description=(
                "Reduce temperature and speed up cooling to encourage convergence."
            ),
            operations=(
                ParameterOperation(
                    parameter_name="temperature",
                    operation_type="multiply",
                    value=0.70,
                ),
                ParameterOperation(
                    parameter_name="cooling_rate",
                    operation_type="add",
                    value=-0.01,
                ),
            ),
        ),
    ]