import networkx as nx

from algorithms.fruchterman_reingold import FruchtermanReingoldAlgorithm
from envs import LayoutContext


def main() -> None:
    """
    Step 3 framework test.

    This verifies that the general layout context and history buffer work.
    """

    graph = nx.cycle_graph(10)
    algorithm = FruchtermanReingoldAlgorithm()

    positions = algorithm.initialize_layout(
        graph=graph,
        seed=42,
        layout_scale=1.0,
    )

    parameters = algorithm.default_parameters(
        graph=graph,
        layout_scale=1.0,
    )

    initial_metrics = {
        "crossing_count": 0.0,
        "angular_resolution": 0.5,
    }

    initial_scores = {
        "layout_score": 0.40,
        "crossing_score": 1.0,
        "angular_resolution_score": 0.5,
        "edge_length_score": 0.4,
        "node_separation_score": 0.3,
    }

    action_names = [
        "no_change",
        "small_increase_k",
        "small_decrease_k",
        "large_increase_k",
        "large_decrease_k",
    ]

    context = LayoutContext.create_initial(
        graph=graph,
        algorithm_name=algorithm.algorithm_name,
        positions=positions,
        parameters=parameters,
        layout_scale=1.0,
        max_steps=5,
        metrics=initial_metrics,
        scores=initial_scores,
        layout_stats={},
        action_names=action_names,
        graph_id="cycle_10_test",
    )

    result = algorithm.step(
        graph=graph,
        positions=context.positions,
        parameters=context.parameters,
        iterations=20,
        layout_scale=1.0,
    )

    updated_metrics = {
        "crossing_count": 0.0,
        "angular_resolution": 0.6,
    }

    updated_scores = {
        "layout_score": 0.55,
        "crossing_score": 1.0,
        "angular_resolution_score": 0.6,
        "edge_length_score": 0.5,
        "node_separation_score": 0.4,
    }

    context.update_after_step(
        action_id=1,
        action_name="small_increase_k",
        new_positions=result.positions,
        new_parameters=result.parameters,
        new_metrics=updated_metrics,
        new_scores=updated_scores,
        new_layout_stats=result.stats,
    )

    assert context.current_step == 1
    assert context.previous_positions is not None
    assert context.previous_parameters is not None
    assert context.previous_scores is not None
    assert context.history.size() == 2
    assert context.last_action_name == "small_increase_k"

    delta_score = context.get_score_delta("layout_score")
    assert abs(delta_score - 0.15) < 1e-8

    displacement_summary = context.get_position_displacement_summary()
    assert displacement_summary["context_mean_displacement"] >= 0.0

    one_hot = context.history.last_action_one_hot(num_actions=len(action_names))
    assert one_hot.shape[0] == len(action_names)
    assert one_hot[1] == 1.0

    print("Layout context test passed.")
    print("Context summary:")
    print(context.to_summary_dict())
    print("Score delta:", delta_score)
    print("Displacement summary:", displacement_summary)
    print("History debug:", context.history.to_debug_dict())


if __name__ == "__main__":
    main()