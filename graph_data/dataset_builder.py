from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import networkx as nx

from graph_data.generators import generate_graph
from graph_data.io import MetadataRow, save_graph_pickle, save_metadata_csv
from graph_data.splits import SplitSpec, default_split_specs, sample_node_count


@dataclass(frozen=True)
class DatasetBuildConfig:
    """
    Configuration for graph dataset generation.
    """

    output_graph_dir: str = "data/processed/graphs"
    metadata_path: str = "data/metadata/dataset_metadata.csv"
    base_seed: int = 2026
    overwrite: bool = True


def graph_density(graph: nx.Graph) -> float:
    """
    Compute graph density.
    """
    n = graph.number_of_nodes()

    if n <= 1:
        return 0.0

    possible_edges = n * (n - 1) / 2.0

    return float(graph.number_of_edges() / possible_edges)


def make_graph_id(
    split_name: str,
    family: str,
    size_label: str,
    index: int,
    seed: int,
) -> str:
    """
    Create a stable graph id.
    """
    return (
        f"{split_name}__{family}__{size_label}"
        f"__idx{index:04d}__seed{seed}"
    )


def build_dataset(
    config: DatasetBuildConfig | None = None,
    split_specs: tuple[SplitSpec, ...] | None = None,
) -> List[MetadataRow]:
    """
    Build the graph dataset and save metadata.

    This function creates:

        data/processed/graphs/<split>/<family>/<graph_id>.pkl
        data/metadata/dataset_metadata.csv
    """
    if config is None:
        config = DatasetBuildConfig()

    if split_specs is None:
        split_specs = default_split_specs()

    graph_root = Path(config.output_graph_dir)
    metadata_path = Path(config.metadata_path)

    if config.overwrite:
        if graph_root.exists():
            shutil.rmtree(graph_root)

        if metadata_path.exists():
            metadata_path.unlink()

    graph_root.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[MetadataRow] = []

    for split_spec in split_specs:
        for family in split_spec.families:
            for size_label in split_spec.size_labels:
                for index in range(split_spec.graphs_per_family_size):
                    seed = (
                        config.base_seed
                        + split_spec.seed_offset
                        + index
                        + 10_000 * split_spec.families.index(family)
                        + 1_000 * split_spec.size_labels.index(size_label)
                    )

                    n_requested = sample_node_count(
                        size_label=size_label,
                        seed=seed,
                    )

                    graph = generate_graph(
                        family=family,
                        n=n_requested,
                        seed=seed,
                    )

                    graph_id = make_graph_id(
                        split_name=split_spec.split_name,
                        family=family,
                        size_label=size_label,
                        index=index,
                        seed=seed,
                    )

                    graph.graph["graph_id"] = graph_id
                    graph.graph["split"] = split_spec.split_name
                    graph.graph["family"] = family
                    graph.graph["size_label"] = size_label
                    graph.graph["seed"] = seed
                    graph.graph["n_requested"] = n_requested

                    graph_path = (
                        graph_root
                        / split_spec.split_name
                        / family
                        / f"{graph_id}.pkl"
                    )

                    save_graph_pickle(
                        graph=graph,
                        path=graph_path,
                    )

                    row: MetadataRow = {
                        "graph_id": graph_id,
                        "split": split_spec.split_name,
                        "family": family,
                        "size_label": size_label,
                        "seed": str(seed),
                        "n_requested": str(n_requested),
                        "n": str(graph.number_of_nodes()),
                        "m": str(graph.number_of_edges()),
                        "density": f"{graph_density(graph):.8f}",
                        "graph_path": str(graph_path),
                    }

                    rows.append(row)

    save_metadata_csv(
        rows=rows,
        path=metadata_path,
    )

    print("Dataset generation completed.")
    print(f"Graphs saved under: {graph_root}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total graphs: {len(rows)}")

    print("\nSplit summary:")

    split_names = sorted(set(row["split"] for row in rows))

    for split_name in split_names:
        split_count = sum(1 for row in rows if row["split"] == split_name)
        print(f"  {split_name}: {split_count}")

    return rows


def main() -> None:
    """
    Build the default graph dataset.
    """
    build_dataset(
        config=DatasetBuildConfig(
            output_graph_dir="data/processed/graphs",
            metadata_path="data/metadata/dataset_metadata.csv",
            base_seed=2026,
            overwrite=True,
        )
    )


if __name__ == "__main__":
    main()