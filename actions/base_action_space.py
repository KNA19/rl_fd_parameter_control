from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from gymnasium import spaces

from algorithms.base import ParameterDict, ParameterSpace
from features.normalizers import safe_float


@dataclass(frozen=True)
class ParameterOperation:
    """
    One parameter update operation.

    Supported operation types:

        multiply:
            parameter <- parameter * value

        add:
            parameter <- parameter + value

        reset:
            parameter <- default parameter value
    """

    parameter_name: str
    operation_type: str
    value: float = 0.0


@dataclass(frozen=True)
class ActionSpec:
    """
    Description of one discrete parameter-control action.
    """

    action_id: int
    name: str
    description: str
    operations: Tuple[ParameterOperation, ...]


@dataclass(frozen=True)
class ActionResult:
    """
    Result returned after applying an action.
    """

    action_id: int
    action_name: str
    old_parameters: ParameterDict
    new_parameters: ParameterDict
    changed: bool

    def to_info_dict(self) -> Dict[str, float | int | str]:
        info: Dict[str, float | int | str] = {
            "action_id": self.action_id,
            "action_name": self.action_name,
            "changed": int(self.changed),
        }

        for key, value in self.old_parameters.items():
            info[f"old_param::{key}"] = float(value)

        for key, value in self.new_parameters.items():
            info[f"new_param::{key}"] = float(value)

        return info


class BaseParameterActionSpace:
    """
    General discrete parameter-control action space.

    This class does not move graph nodes directly. It only changes algorithm
    parameters and then the force-directed algorithm performs node movement.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        action_specs: Sequence[ActionSpec],
    ):
        if not action_specs:
            raise ValueError("At least one action specification is required.")

        self.parameter_space = parameter_space
        self.action_specs: Tuple[ActionSpec, ...] = tuple(action_specs)

        expected_ids = list(range(len(self.action_specs)))
        actual_ids = [spec.action_id for spec in self.action_specs]

        if actual_ids != expected_ids:
            raise ValueError(
                "Action IDs must be consecutive integers starting from 0. "
                f"Expected {expected_ids}, got {actual_ids}."
            )

        self.action_names: Tuple[str, ...] = tuple(
            spec.name for spec in self.action_specs
        )

    @property
    def num_actions(self) -> int:
        return len(self.action_specs)

    def make_gym_space(self) -> spaces.Discrete:
        """
        Return Gymnasium discrete action space.
        """
        return spaces.Discrete(self.num_actions)

    def get_action_name(
        self,
        action_id: int,
    ) -> str:
        """
        Return action name by action id.
        """
        action_index = self._validate_action_id(action_id)
        return self.action_specs[action_index].name

    def get_action_spec(
        self,
        action_id: int,
    ) -> ActionSpec:
        """
        Return full action specification.
        """
        action_index = self._validate_action_id(action_id)
        return self.action_specs[action_index]

    def apply(
        self,
        parameters: Mapping[str, float],
        action_id: int,
    ) -> ActionResult:
        """
        Apply an action to the current parameter dictionary.

        The result is clipped using the algorithm's ParameterSpace.
        """
        action_index = self._validate_action_id(action_id)
        action_spec = self.action_specs[action_index]

        old_parameters = self.parameter_space.clip(parameters)
        new_parameters: ParameterDict = dict(old_parameters)

        for operation in action_spec.operations:
            self._apply_operation(
                parameters=new_parameters,
                operation=operation,
            )

        new_parameters = self.parameter_space.clip(new_parameters)

        changed = self._parameters_changed(
            old_parameters=old_parameters,
            new_parameters=new_parameters,
        )

        return ActionResult(
            action_id=action_index,
            action_name=action_spec.name,
            old_parameters=old_parameters,
            new_parameters=new_parameters,
            changed=changed,
        )

    def describe_actions(self) -> Tuple[Dict[str, str | int], ...]:
        """
        Return action descriptions for printing/debugging.
        """
        descriptions = []

        for spec in self.action_specs:
            descriptions.append(
                {
                    "action_id": spec.action_id,
                    "name": spec.name,
                    "description": spec.description,
                }
            )

        return tuple(descriptions)

    def print_action_definition(self) -> None:
        """
        Print readable action-space summary.
        """
        print("Parameter-control action space")
        print("------------------------------")
        print(f"Number of actions: {self.num_actions}")

        for spec in self.action_specs:
            print(f"  {spec.action_id:02d}: {spec.name}")
            print(f"      {spec.description}")

    def _apply_operation(
        self,
        parameters: ParameterDict,
        operation: ParameterOperation,
    ) -> None:
        parameter_name = operation.parameter_name

        if parameter_name not in self.parameter_space.specs:
            raise KeyError(
                f"Unknown parameter '{parameter_name}' for this action space."
            )

        spec = self.parameter_space.specs[parameter_name]
        current_value = safe_float(
            parameters.get(parameter_name, spec.default_value),
            default=spec.default_value,
        )

        if operation.operation_type == "multiply":
            updated_value = current_value * safe_float(operation.value, 1.0)

        elif operation.operation_type == "add":
            updated_value = current_value + safe_float(operation.value, 0.0)

        elif operation.operation_type == "reset":
            updated_value = spec.default_value

        else:
            raise ValueError(
                f"Unsupported operation type: {operation.operation_type}"
            )

        parameters[parameter_name] = spec.clip(updated_value)

    def _validate_action_id(
        self,
        action_id: int,
    ) -> int:
        action_index = int(action_id)

        if action_index < 0 or action_index >= self.num_actions:
            raise ValueError(
                f"Invalid action_id={action_id}. "
                f"Valid range: 0 to {self.num_actions - 1}."
            )

        return action_index

    def _parameters_changed(
        self,
        old_parameters: Mapping[str, float],
        new_parameters: Mapping[str, float],
    ) -> bool:
        for parameter_name in self.parameter_space.parameter_names:
            old_value = safe_float(old_parameters.get(parameter_name, 0.0))
            new_value = safe_float(new_parameters.get(parameter_name, 0.0))

            if abs(old_value - new_value) > 1e-12:
                return True

        return False