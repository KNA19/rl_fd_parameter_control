import networkx as nx
import numpy as np

from algorithms.fruchterman_reingold import FruchtermanReingoldAlgorithm
from envs import LayoutContext
from metrics import LayoutQualityEvaluator, LayoutScoreCalculator
from states import create_state_builder


def main() -> None:
    """
    Step 8 test.

    This verifies the final redesigned state builder:

        s_t = [
            G_h,
            G_e,
            P_t,
            A_t,
            D_t,
            C_t,
            H_t
        ]

    where delta layout-quality features are included inside H_t.
    """

    graph = nx.cycle_graph(12)

    algorithm = FruchtermanReingoldAlgorithm()
    evaluator = LayoutQualityEvaluator()
    score_calculator = LayoutScoreCalculator()

    parameter_space = algorithm.get_parameter_space(
        graph=graph,
        layout_scale=1.0,
    )

    action_names = [
        "no_change",
        "small_increase_k",
        "small_decrease_k",
        "large_increase_k",
        "large_decrease_k",
        "small_increase_temperature",
        "small_decrease_temperature",
        "large_increase_temperature",
        "large_decrease_temperature",
        "small_increase_cooling_rate",
        "small_decrease_cooling_rate",
        "large_increase_cooling_rate",
        "large_decrease_cooling_rate",
        "expand_and_explore",
        "expand_and_stabilize",
        "compress_and_stabilize",
        "reheat_layout",
        "cool_down_layout",
        "reset_k_to_default",
    ]

    positions = algorithm.initialize_layout(
        graph=graph,
        seed=42,
        layout_scale=1.0,
    )

    parameters = algorithm.default_parameters(
        graph=graph,
        layout_scale=1.0,
    )

    initial_metrics = evaluator.evaluate(
        graph=graph,
        positions=positions,
    )

    initial_scores = score_calculator.score(
        metrics=initial_metrics,
    )

    context = LayoutContext.create_initial(
        graph=graph,
        algorithm_name=algorithm.algorithm_name,
        positions=positions,
        parameters=parameters,
        layout_scale=1.0,
        max_steps=10,
        metrics=initial_metrics,
        scores=initial_scores,
        layout_stats={},
        action_names=action_names,
        graph_id="cycle_12_state_builder_test",
    )

    first_result = algorithm.step(
        graph=graph,
        positions=context.positions,
        parameters=context.parameters,
        iterations=20,
        layout_scale=1.0,
    )

    first_metrics = evaluator.evaluate(
        graph=graph,
        positions=first_result.positions,
    )

    first_scores = score_calculator.score(
        metrics=first_metrics,
    )

    context.update_after_step(
        action_id=1,
        action_name="small_increase_k",
        new_positions=first_result.positions,
        new_parameters=first_result.parameters,
        new_metrics=first_metrics,
        new_scores=first_scores,
        new_layout_stats=first_result.stats,
    )

    second_result = algorithm.step(
        graph=graph,
        positions=context.positions,
        parameters=context.parameters,
        iterations=20,
        layout_scale=1.0,
    )

    second_metrics = evaluator.evaluate(
        graph=graph,
        positions=second_result.positions,
    )

    second_scores = score_calculator.score(
        metrics=second_metrics,
    )

    context.update_after_step(
        action_id=1,
        action_name="small_increase_k",
        new_positions=second_result.positions,
        new_parameters=second_result.parameters,
        new_metrics=second_metrics,
        new_scores=second_scores,
        new_layout_stats=second_result.stats,
    )

    state_builder = create_state_builder(
        state_name="full",
        parameter_space=parameter_space,
        action_names=action_names,
    )

    observation = state_builder.build(
        context=context,
        parameter_space=parameter_space,
    )

    observation_space = state_builder.make_observation_space()

    assert observation.shape[0] == state_builder.observation_dim
    assert observation_space.shape == (state_builder.observation_dim,)
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= 0.0)
    assert np.all(observation <= 1.0)

    component_vectors = state_builder.build_component_vectors(
        context=context,
        parameter_space=parameter_space,
    )

    for component_name, vector in component_vectors.items():
        assert np.all(np.isfinite(vector))
        assert np.all(vector >= 0.0)
        assert np.all(vector <= 1.0)

        print(
            f"{component_name}: "
            f"dim={vector.shape[0]}, "
            f"min={float(np.min(vector)):.6f}, "
            f"max={float(np.max(vector)):.6f}"
        )

    print("\nState builder test passed.")
    print("\nState definition:")
    state_builder.print_state_definition()

    print("\nObservation summary:")
    print(f"  shape: {observation.shape}")
    print(f"  min: {float(np.min(observation)):.6f}")
    print(f"  max: {float(np.max(observation)):.6f}")
    print(f"  mean: {float(np.mean(observation)):.6f}")

    print("\nComponent dimensions:")
    print(state_builder.component_dimensions())

    print("\nSchema summary:")
    print(state_builder.schema.to_summary_dict())


if __name__ == "__main__":
    main()