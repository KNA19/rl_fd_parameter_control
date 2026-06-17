import networkx as nx
import numpy as np

from actions import create_action_space
from algorithms.fruchterman_reingold import FruchtermanReingoldAlgorithm


def main() -> None:
    """
    Step 9 test.

    This verifies that the redesigned multi-scale parameter-control action
    space works with Pure FR.
    """
    graph = nx.cycle_graph(12)

    algorithm = FruchtermanReingoldAlgorithm()

    parameter_space = algorithm.get_parameter_space(
        graph=graph,
        layout_scale=1.0,
    )

    parameters = algorithm.default_parameters(
        graph=graph,
        layout_scale=1.0,
    )

    action_space = create_action_space(
        action_space_name="pure_fr_multiscale",
        parameter_space=parameter_space,
        algorithm_name=algorithm.algorithm_name,
    )

    gym_space = action_space.make_gym_space()

    assert gym_space.n == action_space.num_actions
    assert action_space.num_actions > 0
    assert "no_change" in action_space.action_names
    assert "reset_k_to_default" in action_space.action_names
    assert "expand_and_explore" in action_space.action_names
    assert "reheat_layout" in action_space.action_names

    print("Action-space test")
    print("-----------------")
    print(f"Number of actions: {action_space.num_actions}")
    print("Parameter names:", parameter_space.parameter_names)
    print("Initial parameters:", parameters)

    for action_id in range(action_space.num_actions):
        result = action_space.apply(
            parameters=parameters,
            action_id=action_id,
        )

        normalized = parameter_space.normalize(result.new_parameters)

        assert set(result.new_parameters.keys()) == set(
            parameter_space.parameter_names
        )

        assert np.all(np.isfinite(normalized))
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

        for parameter_name, value in result.new_parameters.items():
            spec = parameter_space.specs[parameter_name]
            assert spec.min_value <= value <= spec.max_value

        print(
            f"{action_id:02d} | "
            f"{result.action_name:35s} | "
            f"changed={result.changed} | "
            f"{result.new_parameters}"
        )

    print("\nAction definitions:")
    action_space.print_action_definition()

    print("\nStep 9 action-space test passed.")


if __name__ == "__main__":
    main()