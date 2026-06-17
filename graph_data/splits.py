from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


SizeRangeDict = Dict[str, Tuple[int, int]]


SIZE_RANGES: SizeRangeDict = {
    "small": (10, 30),
    "medium": (31, 80),
    "large": (81, 150),
    "extra_large": (151, 250),
}


TRAIN_FAMILIES: Tuple[str, ...] = (
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
    "tree",
    "grid",
    "bipartite",
)


UNSEEN_FAMILIES: Tuple[str, ...] = (
    "random_geometric",
    "clustered",
)


@dataclass(frozen=True)
class SplitSpec:
    """
    Dataset split specification.
    """

    split_name: str
    families: Tuple[str, ...]
    size_labels: Tuple[str, ...]
    graphs_per_family_size: int
    seed_offset: int


def default_split_specs() -> Tuple[SplitSpec, ...]:
    """
    Default dataset split plan.

    train:
        seen families and seen sizes

    val:
        seen families and seen sizes

    test_seen:
        seen families and seen sizes, but different random seeds

    test_unseen_size:
        seen families but larger graph sizes

    test_unseen_family:
        unseen families across small/medium/large sizes
    """
    return (
        SplitSpec(
            split_name="train",
            families=TRAIN_FAMILIES,
            size_labels=("small", "medium"),
            graphs_per_family_size=8,
            seed_offset=100_000,
        ),
        SplitSpec(
            split_name="val",
            families=TRAIN_FAMILIES,
            size_labels=("small", "medium"),
            graphs_per_family_size=3,
            seed_offset=200_000,
        ),
        SplitSpec(
            split_name="test_seen",
            families=TRAIN_FAMILIES,
            size_labels=("small", "medium"),
            graphs_per_family_size=3,
            seed_offset=300_000,
        ),
        SplitSpec(
            split_name="test_unseen_size",
            families=TRAIN_FAMILIES,
            size_labels=("large",),
            graphs_per_family_size=3,
            seed_offset=400_000,
        ),
        SplitSpec(
            split_name="test_unseen_family",
            families=UNSEEN_FAMILIES,
            size_labels=("small", "medium", "large"),
            graphs_per_family_size=4,
            seed_offset=500_000,
        ),
    )


def sample_node_count(
    size_label: str,
    seed: int,
) -> int:
    """
    Sample node count from a size label.
    """
    if size_label not in SIZE_RANGES:
        available = ", ".join(sorted(SIZE_RANGES.keys()))
        raise ValueError(
            f"Unknown size label: {size_label}. "
            f"Available labels: {available}"
        )

    low, high = SIZE_RANGES[size_label]

    rng = np.random.default_rng(seed)

    return int(rng.integers(low, high + 1))