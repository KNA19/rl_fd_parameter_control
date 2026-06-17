from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes

from visualization.layout_runner import PolicyLayoutRunResult


PositionDict = Mapping[object, np.ndarray]


def plot_policy_comparison(
    results: List[PolicyLayoutRunResult],
    output_path: str,
    title: str = "Policy Layout Comparison",
) -> None:
    """
    Plot initial layout and final layouts from multiple policies.

    The first panel is the initial layout.
    Each next panel is one policy result.
    """
    if not results:
        raise ValueError("At least one result is required for plotting.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reference = results[0]

    num_panels = 1 + len(results)

    fig_width = max(5.0 * num_panels, 12.0)
    fig_height = 5.5

    fig, axes = plt.subplots(
        1,
        num_panels,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    axes_row = axes[0]

    _draw_graph_layout(
        axis=axes_row[0],
        graph=reference.graph,
        positions=reference.initial_positions,
        title_text=(
            "Initial\n"
            f"score={reference.initial_score:.4f}"
        ),
    )

    for index, result in enumerate(results, start=1):
        action_summary = _summarize_actions(result.action_sequence)

        panel_title = (
            f"{result.policy_name}\n"
            f"final={result.final_score:.4f}, "
            f"Δ={result.improvement:+.4f}\n"
            f"{action_summary}"
        )

        _draw_graph_layout(
            axis=axes_row[index],
            graph=result.graph,
            positions=result.final_positions,
            title_text=panel_title,
        )

    graph_info = (
        f"{title}\n"
        f"Graph: {reference.graph_id} | "
        f"Family: {reference.family} | "
        f"Size: {reference.size_label} | "
        f"n={reference.graph.number_of_nodes()}, "
        f"m={reference.graph.number_of_edges()}"
    )

    fig.suptitle(graph_info, fontsize=12)

    plt.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visual comparison to: {output_file}")


def _draw_graph_layout(
    axis: Axes,
    graph: nx.Graph,
    positions: PositionDict,
    title_text: str,
) -> None:
    """
    Draw one graph layout on one Matplotlib axis.
    """
    axis.set_title(title_text, fontsize=9)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    pos_2d = {
        node: (
            float(np.asarray(position)[0]),
            float(np.asarray(position)[1]),
        )
        for node, position in positions.items()
    }

    nx.draw_networkx_edges(
        graph,
        pos=pos_2d,
        ax=axis,
        width=0.8,
        alpha=0.6,
    )

    nx.draw_networkx_nodes(
        graph,
        pos=pos_2d,
        ax=axis,
        node_size=45,
        linewidths=0.4,
        edgecolors="black",
    )

    _set_equal_axis_limits(
        axis=axis,
        positions=pos_2d,
    )


def _set_equal_axis_limits(
    axis: Axes,
    positions: Mapping[object, tuple[float, float]],
) -> None:
    """
    Set equal-looking axis limits with padding.
    """
    if not positions:
        return

    xs = [value[0] for value in positions.values()]
    ys = [value[1] for value in positions.values()]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    span = max(x_span, y_span)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    padding = 0.15 * span

    axis.set_xlim(
        x_center - 0.5 * span - padding,
        x_center + 0.5 * span + padding,
    )

    axis.set_ylim(
        y_center - 0.5 * span - padding,
        y_center + 0.5 * span + padding,
    )


def _summarize_actions(
    action_sequence: List[str],
    max_items: int = 2,
) -> str:
    """
    Return compact action summary for subplot title.
    """
    if not action_sequence:
        return "actions: none"

    counts = Counter(action_sequence)
    most_common = counts.most_common(max_items)

    parts = [f"{name}×{count}" for name, count in most_common]

    return "actions: " + ", ".join(parts)