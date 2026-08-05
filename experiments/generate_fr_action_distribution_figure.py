from __future__ import annotations

from visualization.ppo_action_distribution_plot import (
    generate_fr_sarl_action_distribution_figure,
)


def main() -> None:
    """
    Generate the thesis figure:
    100% stacked horizontal bar chart of PPO action distribution
    across the three test conditions.
    """
    generate_fr_sarl_action_distribution_figure()


if __name__ == "__main__":
    main()