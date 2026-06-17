from pathlib import Path

from graph_data import DatasetBuildConfig, build_dataset, load_graph_pickle


def main() -> None:
    """
    Step 5 test.

    This builds a small graph dataset using the default split design and
    verifies that graph files and metadata are created correctly.
    """
    test_graph_dir = "data/processed/test_graphs"
    test_metadata_path = "data/metadata/test_dataset_metadata.csv"

    rows = build_dataset(
        config=DatasetBuildConfig(
            output_graph_dir=test_graph_dir,
            metadata_path=test_metadata_path,
            base_seed=123,
            overwrite=True,
        )
    )

    assert len(rows) > 0

    metadata_file = Path(test_metadata_path)
    assert metadata_file.exists()

    first_graph_path = rows[0]["graph_path"]
    graph = load_graph_pickle(first_graph_path)

    assert graph.number_of_nodes() >= 2
    assert graph.number_of_edges() >= 1
    assert "graph_id" in graph.graph
    assert "family" in graph.graph
    assert "split" in graph.graph

    splits = set(row["split"] for row in rows)

    expected_splits = {
        "train",
        "val",
        "test_seen",
        "test_unseen_size",
        "test_unseen_family",
    }

    assert expected_splits.issubset(splits)

    print("Graph-data test passed.")
    print(f"Generated metadata: {test_metadata_path}")
    print(f"Number of generated graphs: {len(rows)}")
    print(f"First graph path: {first_graph_path}")
    print(f"First graph nodes: {graph.number_of_nodes()}")
    print(f"First graph edges: {graph.number_of_edges()}")
    print(f"Splits found: {sorted(splits)}")


if __name__ == "__main__":
    main()