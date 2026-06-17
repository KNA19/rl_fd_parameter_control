import networkx as nx
import numpy as np

from features import GraphFeatureExtractor, SpectralGraphEmbedding
from graph_data.generators import available_graph_families, generate_graph


def test_single_graph_features() -> None:
    """
    Test graph features and spectral embedding on one graph.
    """
    graph = nx.cycle_graph(10)

    feature_extractor = GraphFeatureExtractor()
    embedding_extractor = SpectralGraphEmbedding()

    features = feature_extractor.extract(graph)
    feature_vector = feature_extractor.to_vector(graph)

    embedding = embedding_extractor.extract(graph)
    embedding_vector = embedding_extractor.to_vector(graph)

    assert len(features) == feature_extractor.feature_dim
    assert feature_vector.shape[0] == feature_extractor.feature_dim
    assert len(embedding) == embedding_extractor.embedding_dim
    assert embedding_vector.shape[0] == embedding_extractor.embedding_dim

    assert np.all(np.isfinite(feature_vector))
    assert np.all(np.isfinite(embedding_vector))

    assert np.all(feature_vector >= 0.0)
    assert np.all(feature_vector <= 1.0)
    assert np.all(embedding_vector >= 0.0)
    assert np.all(embedding_vector <= 1.0)

    print("Single graph feature test passed.")
    print(f"Feature dimension: {feature_extractor.feature_dim}")
    print(f"Embedding dimension: {embedding_extractor.embedding_dim}")
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.6f}")

    print("Embedding:")
    for key, value in embedding.items():
        print(f"  {key}: {value:.6f}")


def test_all_generated_families() -> None:
    """
    Test feature extraction on all available graph families.
    """
    feature_extractor = GraphFeatureExtractor()
    embedding_extractor = SpectralGraphEmbedding()

    families = available_graph_families()

    print("\nTesting graph families:")

    for index, family in enumerate(families):
        graph = generate_graph(
            family=family,
            n=30,
            seed=2026 + index,
        )

        feature_vector = feature_extractor.to_vector(graph)
        embedding_vector = embedding_extractor.to_vector(graph)

        assert feature_vector.shape[0] == feature_extractor.feature_dim
        assert embedding_vector.shape[0] == embedding_extractor.embedding_dim

        assert np.all(np.isfinite(feature_vector))
        assert np.all(np.isfinite(embedding_vector))

        assert np.all(feature_vector >= 0.0)
        assert np.all(feature_vector <= 1.0)
        assert np.all(embedding_vector >= 0.0)
        assert np.all(embedding_vector <= 1.0)

        print(
            f"  {family}: "
            f"n={graph.number_of_nodes()}, "
            f"m={graph.number_of_edges()}, "
            f"feature_dim={feature_vector.shape[0]}, "
            f"embedding_dim={embedding_vector.shape[0]}"
        )

    print("\nAll graph-family feature tests passed.")


def main() -> None:
    test_single_graph_features()
    test_all_generated_families()


if __name__ == "__main__":
    main()