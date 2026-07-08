from __future__ import annotations

from visualization.test_combined_boxplot import generate_combined_test_boxplot


def main() -> None:
    """
    Generate Figure 1:

    One combined test-class box plot using:
        test_seen
        test_unseen_size
        test_unseen_family

    Policies:
        Default FR
        Random
        Best fixed
        PPO
    """
    generate_combined_test_boxplot()


if __name__ == "__main__":
    main()