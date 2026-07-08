from __future__ import annotations

from visualization.test_combined_final_score_boxplot import (
    generate_figure2_combined_test_final_score,
)


def main() -> None:
    """
    Generate Figure 2:

    One combined test-class box plot using:
        test_seen
        test_unseen_size
        test_unseen_family

    Metric:
        final layout score

    Policies:
        Default FR
        Random
        Best fixed
        PPO
    """
    generate_figure2_combined_test_final_score()


if __name__ == "__main__":
    main()