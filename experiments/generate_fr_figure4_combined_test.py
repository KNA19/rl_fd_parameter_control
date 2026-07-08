from __future__ import annotations

from visualization.test_combined_visual_grid import (
    generate_figure4_combined_test_visual_grid,
)


def main() -> None:
    """
    Generate Figure 4:

    One combined test-class visual layout grid.

    Rows:
        test_seen
        test_unseen_size
        test_unseen_family

    Columns:
        Initial
        Default FR
        Random
        Best fixed
        PPO
    """
    generate_figure4_combined_test_visual_grid()


if __name__ == "__main__":
    main()