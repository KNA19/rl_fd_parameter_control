from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx


MetadataRow = Dict[str, str]


def save_graph_pickle(
    graph: nx.Graph,
    path: str | Path,
) -> None:
    """
    Save a NetworkX graph as a pickle file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(graph, file)


def load_graph_pickle(
    path: str | Path,
) -> nx.Graph:
    """
    Load a NetworkX graph from a pickle file.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Graph file not found: {input_path}")

    with open(input_path, "rb") as file:
        graph = pickle.load(file)

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected nx.Graph, got {type(graph)}")

    return graph


def save_metadata_csv(
    rows: Iterable[MetadataRow],
    path: str | Path,
) -> None:
    """
    Save dataset metadata to CSV.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_list = list(rows)

    if not row_list:
        raise ValueError("Cannot save empty metadata.")

    fieldnames = list(row_list[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in row_list:
            writer.writerow(row)


def load_metadata_csv(
    path: str | Path,
) -> List[MetadataRow]:
    """
    Load dataset metadata from CSV.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {input_path}")

    rows: List[MetadataRow] = []

    with open(input_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(dict(row))

    return rows