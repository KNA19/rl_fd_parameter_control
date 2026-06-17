from __future__ import annotations

from typing import Tuple

import networkx as nx
import numpy as np

from features.normalizers import FloatDict, clip01, dictionary_to_vector, safe_float


class SpectralGraphEmbedding:
    """
    Deterministic compact graph-level embedding.

    This is not a GNN embedding yet.

    It provides a fixed-length structural representation G_e using
    adjacency and Laplacian spectral summaries.

    Later, this file can be extended with a GAT/GNN embedding module.
    """

    EMBEDDING_NAMES: Tuple[str, ...] = (
        "adj_spectral_radius_norm",
        "adj_second_eigen_abs_norm",
        "adj_energy_norm",
        "lap_algebraic_connectivity_norm",
        "lap_largest_eigen_norm",
        "lap_spectral_gap_norm",
        "lap_trace_norm",
        "lap_eigen_entropy_norm",
        "lap_low_eigen_mean_norm",
        "lap_high_eigen_mean_norm",
        "adj_positive_eigen_fraction",
        "adj_negative_eigen_fraction",
    )

    def __init__(
        self,
        max_nodes: int = 300,
        max_degree: int = 100,
        max_edges: int = 5000,
    ):
        self.max_nodes = max_nodes
        self.max_degree = max_degree
        self.max_edges = max_edges

    @property
    def embedding_dim(self) -> int:
        return len(self.EMBEDDING_NAMES)

    def extract(
        self,
        graph: nx.Graph,
    ) -> FloatDict:
        """
        Return a normalized spectral graph embedding dictionary.
        """
        simple_graph = nx.Graph(graph)
        simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))

        n = simple_graph.number_of_nodes()

        if n == 0:
            return self._empty_embedding()

        adjacency = to_float_array(nx.to_numpy_array(simple_graph))

        degrees = to_float_array(
            [degree for _, degree in simple_graph.degree()]
        )

        laplacian = np.diag(degrees) - adjacency

        try:
            adjacency_eigenvalues = np.linalg.eigvalsh(adjacency)
        except np.linalg.LinAlgError:
            adjacency_eigenvalues = np.zeros(n).astype("float64")

        try:
            laplacian_eigenvalues = np.linalg.eigvalsh(laplacian)
        except np.linalg.LinAlgError:
            laplacian_eigenvalues = np.zeros(n).astype("float64")

        adjacency_abs = np.abs(adjacency_eigenvalues)
        adjacency_abs_sorted = np.sort(adjacency_abs)[::-1]

        adj_spectral_radius = (
            float(adjacency_abs_sorted[0])
            if adjacency_abs_sorted.size > 0
            else 0.0
        )

        adj_second_eigen_abs = (
            float(adjacency_abs_sorted[1])
            if adjacency_abs_sorted.size > 1
            else 0.0
        )

        adj_energy = float(np.sum(adjacency_abs))

        lap_sorted = np.sort(np.maximum(laplacian_eigenvalues, 0.0))

        algebraic_connectivity = (
            float(lap_sorted[1]) if lap_sorted.size > 1 else 0.0
        )

        lap_largest = float(lap_sorted[-1]) if lap_sorted.size > 0 else 0.0

        lap_spectral_gap = (
            float(lap_sorted[2] - lap_sorted[1])
            if lap_sorted.size > 2
            else 0.0
        )

        lap_trace = float(np.sum(lap_sorted))

        lap_entropy = self._eigen_entropy(lap_sorted)

        low_mean, high_mean = self._low_high_laplacian_means(lap_sorted)

        if adjacency_eigenvalues.size == 0:
            positive_fraction = 0.0
            negative_fraction = 0.0
        else:
            positive_fraction = float(np.mean(adjacency_eigenvalues > 1e-9))
            negative_fraction = float(np.mean(adjacency_eigenvalues < -1e-9))

        max_lap_eigen = max(1.0, 2.0 * self.max_degree)
        max_adj_energy = max(1.0, 2.0 * self.max_edges)

        embedding: FloatDict = {
            "adj_spectral_radius_norm": clip01(
                adj_spectral_radius / max(1.0, self.max_degree)
            ),
            "adj_second_eigen_abs_norm": clip01(
                adj_second_eigen_abs / max(1.0, self.max_degree)
            ),
            "adj_energy_norm": clip01(adj_energy / max_adj_energy),
            "lap_algebraic_connectivity_norm": clip01(
                algebraic_connectivity / max_lap_eigen
            ),
            "lap_largest_eigen_norm": clip01(lap_largest / max_lap_eigen),
            "lap_spectral_gap_norm": clip01(lap_spectral_gap / max_lap_eigen),
            "lap_trace_norm": clip01(
                lap_trace / max(1.0, 2.0 * self.max_edges)
            ),
            "lap_eigen_entropy_norm": clip01(lap_entropy),
            "lap_low_eigen_mean_norm": clip01(low_mean / max_lap_eigen),
            "lap_high_eigen_mean_norm": clip01(high_mean / max_lap_eigen),
            "adj_positive_eigen_fraction": clip01(positive_fraction),
            "adj_negative_eigen_fraction": clip01(negative_fraction),
        }

        return self._ordered_embedding_dict(embedding)

    def to_vector(
        self,
        graph: nx.Graph,
    ) -> np.ndarray:
        """
        Return spectral embedding as a fixed-order float32 vector.
        """
        embedding = self.extract(graph)

        return dictionary_to_vector(
            data=embedding,
            names=self.EMBEDDING_NAMES,
        )

    def _empty_embedding(self) -> FloatDict:
        return {name: 0.0 for name in self.EMBEDDING_NAMES}

    def _ordered_embedding_dict(
        self,
        embedding: FloatDict,
    ) -> FloatDict:
        return {
            name: safe_float(embedding.get(name, 0.0))
            for name in self.EMBEDDING_NAMES
        }

    def _eigen_entropy(
        self,
        eigenvalues: np.ndarray,
    ) -> float:
        """
        Normalized entropy of non-negative eigenvalues.
        """
        values = to_float_array(eigenvalues)
        values = np.maximum(values, 0.0)

        total = float(np.sum(values))

        if total <= 1e-12:
            return 0.0

        probabilities = values / total
        probabilities = probabilities[probabilities > 0.0]

        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        max_entropy = np.log(max(2, values.size))

        if max_entropy <= 1e-12:
            return 0.0

        return clip01(entropy / max_entropy)

    def _low_high_laplacian_means(
        self,
        laplacian_eigenvalues: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Return mean of lower and upper halves of Laplacian spectrum.
        """
        values = to_float_array(laplacian_eigenvalues)

        if values.size == 0:
            return 0.0, 0.0

        sorted_values = np.sort(values)
        midpoint = max(1, values.size // 2)

        low_mean = float(np.mean(sorted_values[:midpoint]))

        if midpoint >= values.size:
            high_mean = low_mean
        else:
            high_mean = float(np.mean(sorted_values[midpoint:]))

        return low_mean, high_mean


def to_float_array(value: object) -> np.ndarray:
    """
    Convert input to a float64 NumPy array without using dtype=... arguments.

    This avoids Pylance dtype false-positive warnings.
    """
    return np.asarray(value).astype("float64")