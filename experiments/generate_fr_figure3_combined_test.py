from __future__ import annotations

from visualization.test_combined_component_boxplots import (
    generate_figure3_combined_test_components,
)


def main() -> None:
    """
    Generate Figure 3:

    One combined test-class component-wise box-plot figure using:
        test_seen
        test_unseen_size
        test_unseen_family

    Components:
        crossing score improvement
        angular-resolution score improvement
        edge-length score improvement
        node-separation score improvement

    Policies:
        Default FR
        Random
        Best fixed
        PPO
    """
    generate_figure3_combined_test_components()


if __name__ == "__main__":
    main()