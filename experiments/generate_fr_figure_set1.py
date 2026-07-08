from __future__ import annotations

from visualization.performance_boxplots import generate_figure_set1


def main() -> None:
    """
    Generate Figure Set 1:

    Layout-score improvement box plots for:
        Default FR
        Random
        Best fixed-action baseline
        PPO

    across:
        val
        test_seen
        test_unseen_size
        test_unseen_family
    """
    generate_figure_set1()


if __name__ == "__main__":
    main()