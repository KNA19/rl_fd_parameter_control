import networkx as nx
import numpy as np

from algorithms.fruchterman_reingold import FruchtermanReingoldAlgorithm


def main() -> None:
    """
    Step 2 framework test.

    This verifies that Pure FR now works through the general algorithm
    interface.
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

    normalized_parameters = parameter_space.normalize(parameters)

    positions = algorithm.initialize_layout(
        graph=graph,
        seed=42,
        layout_scale=1.0,
    )

    result = algorithm.step(
        graph=graph,
        positions=positions,
        parameters=parameters,
        iterations=20,
        layout_scale=1.0,
    )

    assert set(result.positions.keys()) == set(graph.nodes)
    assert set(result.parameters.keys()) == set(parameter_space.parameter_names)
    assert normalized_parameters.shape[0] == len(
        parameter_space.parameter_names
    )
    assert np.all(normalized_parameters >= 0.0)
    assert np.all(normalized_parameters <= 1.0)
    assert result.stats.iterations == 20

    print("Algorithm interface test passed.")
    print(f"Algorithm: {algorithm.algorithm_name}")
    print("Parameter names:", parameter_space.parameter_names)
    print("Default parameters:", parameters)
    print("Normalized parameters:", normalized_parameters)
    print("Updated parameters:", result.parameters)
    print("Layout stats:", result.stats.to_dict())


if __name__ == "__main__":
    main()