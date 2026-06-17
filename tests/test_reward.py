import networkx as nx
import numpy as np

from actions import create_action_space
from algorithms.fruchterman_reingold import FruchtermanReingoldAlgorithm
from envs import LayoutContext
from metrics import LayoutQualityEvaluator, LayoutScoreCalculator
from rewards import create_reward_function


def main() -> None:
    """
    Step 10 test.

    This verifies the redesigned reward layer:

        balanced aesthetic improvement
        - expansion penalty
        - repeated-action penalty
        - action-change penalty
        + terminal bonus
    """
    graph = nx.cycle_graph(12)

    algorithm = FruchtermanReingoldAlgorithm()
    evaluator = LayoutQualityEvaluator()
    score_calculator = LayoutScoreCalculator()

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

    positions = algorithm.initialize_layout(
        graph=graph,
        seed=42,
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
        max_steps=3,
        metrics=initial_metrics,
        scores=initial_scores,
        layout_stats={},
        action_names=list(action_space.action_names),
        graph_id="cycle_12_reward_test",
    )

    reward_function = create_reward_function("aesthetic_delta")

    print("Reward test")
    print("-----------")
    print(f"Initial layout score: {initial_scores['layout_score']:.6f}")

    last_action_result = None

    for step_index in range(3):
        action_id = 1  # small_increase_k, repeated intentionally for testing.

        action_result = action_space.apply(
            parameters=context.parameters,
            action_id=action_id,
        )

        layout_result = algorithm.step(
            graph=graph,
            positions=context.positions,
            parameters=action_result.new_parameters,
            iterations=20,
            layout_scale=1.0,
        )

        new_metrics = evaluator.evaluate(
            graph=graph,
            positions=layout_result.positions,
        )

        new_scores = score_calculator.score(
            metrics=new_metrics,
        )

        context.update_after_step(
            action_id=action_result.action_id,
            action_name=action_result.action_name,
            new_positions=layout_result.positions,
            new_parameters=layout_result.parameters,
            new_metrics=new_metrics,
            new_scores=new_scores,
            new_layout_stats=layout_result.stats,
        )

        is_terminal = context.is_terminal

        reward_result = reward_function.compute(
            context=context,
            action_result=action_result,
            is_terminal=is_terminal,
        )

        assert np.isfinite(reward_result.reward)

        for value in reward_result.components.values():
            assert np.isfinite(value)

        print(
            f"\nStep {step_index + 1}"
            f"\n  action: {action_result.action_name}"
            f"\n  layout score: {new_scores['layout_score']:.6f}"
            f"\n  delta layout score: "
            f"{context.get_score_delta('layout_score', 0.0):.6f}"
            f"\n  reward: {reward_result.reward:.6f}"
            f"\n  terminal: {is_terminal}"
        )

        print("  reward components:")

        for key, value in reward_result.components.items():
            print(f"    {key}: {value:.6f}")

        last_action_result = action_result

    assert context.is_terminal
    assert last_action_result is not None

    terminal_reward = reward_function.compute(
        context=context,
        action_result=last_action_result,
        is_terminal=True,
    )

    assert terminal_reward.components["terminal_bonus"] > 0.0

    score_only_reward = create_reward_function("score_only")
    score_only_result = score_only_reward.compute(
        context=context,
        action_result=last_action_result,
        is_terminal=True,
    )

    assert np.isfinite(score_only_result.reward)

    print("\nTerminal reward check passed.")
    print(f"Terminal reward: {terminal_reward.reward:.6f}")
    print(f"Score-only reward: {score_only_result.reward:.6f}")

    print("\nStep 10 reward test passed.")


if __name__ == "__main__":
    main()