from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class StateComponentSchema:
    """
    Describes one component of the final observation vector.

    Example components:
        graph_features
        graph_embedding
        parameter_features
        layout_features
        dynamics_features
        conflict_features
        history_features
    """

    name: str
    feature_names: Tuple[str, ...]
    start_index: int
    end_index: int

    @property
    def dimension(self) -> int:
        return self.end_index - self.start_index


@dataclass(frozen=True)
class StateSchema:
    """
    Full state schema for the redesigned SARL observation vector.

    This keeps track of component names, feature names, dimensions,
    and index ranges in the final vector.
    """

    components: Tuple[StateComponentSchema, ...]

    @property
    def observation_dim(self) -> int:
        if not self.components:
            return 0

        return self.components[-1].end_index

    @property
    def feature_names(self) -> Tuple[str, ...]:
        names: List[str] = []

        for component in self.components:
            for feature_name in component.feature_names:
                names.append(f"{component.name}::{feature_name}")

        return tuple(names)

    def component_dimensions(self) -> Dict[str, int]:
        return {
            component.name: component.dimension
            for component in self.components
        }

    def component_ranges(self) -> Dict[str, Tuple[int, int]]:
        return {
            component.name: (
                component.start_index,
                component.end_index,
            )
            for component in self.components
        }

    def to_summary_dict(self) -> Dict[str, int]:
        summary: Dict[str, int] = {
            "observation_dim": self.observation_dim,
        }

        for component in self.components:
            summary[f"{component.name}_dim"] = component.dimension

        return summary

    @classmethod
    def from_components(
        cls,
        component_features: Mapping[str, Sequence[str]],
    ) -> "StateSchema":
        """
        Build schema from component name to feature-name list.
        """
        components: List[StateComponentSchema] = []
        start_index = 0

        for component_name, feature_names_raw in component_features.items():
            feature_names = tuple(str(name) for name in feature_names_raw)
            end_index = start_index + len(feature_names)

            components.append(
                StateComponentSchema(
                    name=str(component_name),
                    feature_names=feature_names,
                    start_index=start_index,
                    end_index=end_index,
                )
            )

            start_index = end_index

        return cls(components=tuple(components))

    def print_summary(self) -> None:
        """
        Print readable state schema summary.
        """
        print("State schema")
        print("------------")
        print(f"Total observation dimension: {self.observation_dim}")

        for component in self.components:
            print(
                f"  {component.name}: "
                f"dim={component.dimension}, "
                f"range=[{component.start_index}, {component.end_index})"
            )