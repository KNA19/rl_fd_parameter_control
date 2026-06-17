import networkx as nx
import numpy as np

from algorithms.fruchterman_reingold import FruchtermanReingoldAlgorithm
from envs import LayoutContext
from features import (
    ConflictFeatureExtractor,
    DynamicsFeatureExtractor,
    HistoryFeatureExtractor,
    LayoutFeatureExtractor,
)
from metrics import LayoutQualityEvaluator, LayoutScoreCalculator


def main() -> None:
    """
    Step 7 test.

    This verifies that dynamic layout, conflict, dynamics, and history features
    can be extracted from LayoutContext.
    """
    graph = nx.cycle_graph(12)

    algorithm = FruchtermanReingoldAlgorithm()
    evaluator = LayoutQualityEvaluator()
    score_calculator = LayoutScoreCalculator()

    action_names = [
        "no_change",
        "small_increase_k",
        "small_decrease_k",
        "large_increase_k",
        "large_decrease_k",
        "small_increase_temperature",
        "small_decrease_temperature",
        "small_increase_cooling_rate",
        "small_decrease_cooling_rate",
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
        max_steps=5,
        metrics=initial_metrics,
        scores=initial_scores,
        layout_stats={},
        action_names=action_names,
        graph_id="cycle_12_dynamic_feature_test",
    )

    # First update.
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

    # Second update to make deltas/history meaningful.
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

    layout_extractor = LayoutFeatureExtractor()
    dynamics_extractor = DynamicsFeatureExtractor()
    conflict_extractor = ConflictFeatureExtractor()
    history_extractor = HistoryFeatureExtractor(
        action_names=action_names,
        history_window=5,
    )

    layout_vector = layout_extractor.to_vector(context)
    dynamics_vector = dynamics_extractor.to_vector(context)
    conflict_vector = conflict_extractor.to_vector(context)
    history_vector = history_extractor.to_vector(context)

    for name, vector in [
        ("layout", layout_vector),
        ("dynamics", dynamics_vector),
        ("conflict", conflict_vector),
        ("history", history_vector),
    ]:
        assert np.all(np.isfinite(vector))
        assert np.all(vector >= 0.0)
        assert np.all(vector <= 1.0)

        print(f"{name} feature vector passed.")
        print(f"  dimension: {vector.shape[0]}")
        print(f"  min: {float(np.min(vector)):.6f}")
        print(f"  max: {float(np.max(vector)):.6f}")

    print("\nDynamic feature test passed.")
    print("Context summary:")
    print(context.to_summary_dict())

    print("\nFeature dimensions:")
    print(f"  Layout features: {layout_extractor.feature_dim}")
    print(f"  Dynamics features: {dynamics_extractor.feature_dim}")
    print(f"  Conflict features: {conflict_extractor.feature_dim}")
    print(f"  History features: {history_extractor.feature_dim}")


if __name__ == "__main__":
    main()