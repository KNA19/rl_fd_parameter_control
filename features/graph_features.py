from __future__ import annotations

import math
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from features.normalizers import (
    FloatDict,
    clip01,
    dictionary_to_vector,
    normalize_linear,
    normalize_log1p,
    normalize_signed_unit,
    safe_float,
)


class GraphFeatureExtractor:
    """
    Extract fixed-length handcrafted graph descriptors.

    These features represent the static graph context G_h in the redesigned
    state space.

    The extractor is graph-size-invariant and node-label-independent.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "n_norm",
        "m_norm",
        "log_n_norm",
        "log_m_norm",
        "density",
        "edge_node_ratio_norm",
        "degree_mean_norm",
        "degree_std_norm",
        "degree_cv_norm",
        "degree_gini",
        "degree_entropy_norm",
        "leaf_fraction",
        "hub_fraction",
        "avg_clustering",
        "transitivity",
        "avg_shortest_path_norm",
        "diameter_norm",
        "global_efficiency",
        "cycle_rank_norm",
        "is_tree",
        "is_bipartite",
        "is_planar",
        "num_components_norm",
        "largest_component_fraction",
        "core_number_max_norm",
        "core_number_mean_norm",
        "degree_assortativity_norm",
        "modularity_norm",
    )

    def __init__(
        self,
        max_nodes: int = 300,
        max_edges: int = 5000,
        max_degree: int = 100,
        max_diameter: int = 300,
        max_components: int = 20,
        max_core_number: int = 100,
    ):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_degree = max_degree
        self.max_diameter = max_diameter
        self.max_components = max_components
        self.max_core_number = max_core_number

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def extract(
        self,
        graph: nx.Graph,
    ) -> FloatDict:
        """
        Return normalized handcrafted graph descriptors.
        """
        simple_graph = nx.Graph(graph)
        simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))

        n = simple_graph.number_of_nodes()
        m = simple_graph.number_of_edges()

        if n == 0:
            return self._empty_features()

        degrees = np.asarray(
            [degree for _, degree in simple_graph.degree()],
            dtype=float,
        )

        degree_mean = float(np.mean(degrees)) if degrees.size else 0.0
        degree_std = float(np.std(degrees)) if degrees.size else 0.0

        if degree_mean <= 1e-12:
            degree_cv = 0.0
        else:
            degree_cv = degree_std / degree_mean

        leaf_fraction = float(np.mean(degrees == 1.0)) if degrees.size else 0.0

        hub_threshold = max(2.0 * degree_mean, degree_mean + degree_std)

        if degrees.size:
            hub_fraction = float(np.mean(degrees >= hub_threshold))
        else:
            hub_fraction = 0.0

        density = float(nx.density(simple_graph)) if n > 1 else 0.0

        edge_node_ratio = m / max(1, n)

        degree_gini = self._degree_gini(degrees)
        degree_entropy = self._degree_entropy(degrees)

        avg_clustering = self._safe_average_clustering(simple_graph)
        transitivity = self._safe_transitivity(simple_graph)

        num_components = nx.number_connected_components(simple_graph)

        if num_components == 0:
            largest_component_fraction = 0.0
            connected_graph = simple_graph
        else:
            largest_component_nodes = max(
                nx.connected_components(simple_graph),
                key=len,
            )
            largest_component_fraction = len(largest_component_nodes) / max(1, n)
            connected_graph = simple_graph.subgraph(largest_component_nodes).copy()

        avg_shortest_path = self._safe_average_shortest_path_length(
            connected_graph
        )
        diameter = self._safe_diameter(connected_graph)
        global_efficiency = self._safe_global_efficiency(simple_graph)

        cycle_rank = max(0, m - n + num_components)

        is_tree = 1.0 if nx.is_tree(simple_graph) else 0.0
        is_bipartite = 1.0 if nx.is_bipartite(simple_graph) else 0.0
        is_planar = 1.0 if self._safe_is_planar(simple_graph) else 0.0

        core_values = self._core_number_values(simple_graph)

        if core_values.size == 0:
            core_max = 0.0
            core_mean = 0.0
        else:
            core_max = float(np.max(core_values))
            core_mean = float(np.mean(core_values))

        assortativity = self._safe_degree_assortativity(simple_graph)
        modularity = self._safe_modularity(simple_graph)

        features: FloatDict = {
            "n_norm": normalize_linear(n, self.max_nodes),
            "m_norm": normalize_linear(m, self.max_edges),
            "log_n_norm": normalize_log1p(n, self.max_nodes),
            "log_m_norm": normalize_log1p(m, self.max_edges),
            "density": clip01(density),
            "edge_node_ratio_norm": normalize_linear(edge_node_ratio, 20.0),
            "degree_mean_norm": normalize_linear(degree_mean, self.max_degree),
            "degree_std_norm": normalize_linear(degree_std, self.max_degree),
            "degree_cv_norm": normalize_linear(degree_cv, 5.0),
            "degree_gini": clip01(degree_gini),
            "degree_entropy_norm": clip01(degree_entropy),
            "leaf_fraction": clip01(leaf_fraction),
            "hub_fraction": clip01(hub_fraction),
            "avg_clustering": clip01(avg_clustering),
            "transitivity": clip01(transitivity),
            "avg_shortest_path_norm": normalize_linear(
                avg_shortest_path,
                self.max_diameter,
            ),
            "diameter_norm": normalize_linear(diameter, self.max_diameter),
            "global_efficiency": clip01(global_efficiency),
            "cycle_rank_norm": normalize_linear(cycle_rank, max(1, self.max_edges)),
            "is_tree": clip01(is_tree),
            "is_bipartite": clip01(is_bipartite),
            "is_planar": clip01(is_planar),
            "num_components_norm": normalize_linear(
                num_components,
                self.max_components,
            ),
            "largest_component_fraction": clip01(largest_component_fraction),
            "core_number_max_norm": normalize_linear(
                core_max,
                self.max_core_number,
            ),
            "core_number_mean_norm": normalize_linear(
                core_mean,
                self.max_core_number,
            ),
            "degree_assortativity_norm": normalize_signed_unit(assortativity),
            "modularity_norm": clip01(modularity),
        }

        return self._ordered_feature_dict(features)

    def to_vector(
        self,
        graph: nx.Graph,
    ) -> np.ndarray:
        """
        Return graph features as a fixed-order float32 vector.
        """
        features = self.extract(graph)

        return dictionary_to_vector(
            data=features,
            names=self.FEATURE_NAMES,
        )

    def _ordered_feature_dict(
        self,
        features: FloatDict,
    ) -> FloatDict:
        return {
            name: safe_float(features.get(name, 0.0))
            for name in self.FEATURE_NAMES
        }

    def _empty_features(self) -> FloatDict:
        return {name: 0.0 for name in self.FEATURE_NAMES}

    def _degree_gini(
        self,
        degrees: np.ndarray,
    ) -> float:
        """
        Compute Gini coefficient of degree distribution.
        """
        if degrees.size == 0:
            return 0.0

        values = np.sort(degrees)

        if np.sum(values) <= 1e-12:
            return 0.0

        n = values.size
        index = np.arange(1, n + 1)

        gini = (
            np.sum((2.0 * index - n - 1.0) * values)
            / (n * np.sum(values))
        )

        return clip01(float(gini))

    def _degree_entropy(
        self,
        degrees: np.ndarray,
    ) -> float:
        """
        Normalized entropy of degree distribution.
        """
        if degrees.size == 0:
            return 0.0

        total = float(np.sum(degrees))

        if total <= 1e-12:
            return 0.0

        probabilities = degrees / total
        probabilities = probabilities[probabilities > 0.0]

        entropy = -float(np.sum(probabilities * np.log(probabilities)))

        max_entropy = math.log(max(2, degrees.size))

        if max_entropy <= 1e-12:
            return 0.0

        return clip01(entropy / max_entropy)

    def _safe_average_clustering(
        self,
        graph: nx.Graph,
    ) -> float:
        try:
            return safe_float(nx.average_clustering(graph))
        except nx.NetworkXException:
            return 0.0

    def _safe_transitivity(
        self,
        graph: nx.Graph,
    ) -> float:
        try:
            return safe_float(nx.transitivity(graph))
        except nx.NetworkXException:
            return 0.0

    def _safe_average_shortest_path_length(
        self,
        graph: nx.Graph,
    ) -> float:
        try:
            if graph.number_of_nodes() <= 1:
                return 0.0

            return safe_float(nx.average_shortest_path_length(graph))
        except nx.NetworkXException:
            return 0.0

    def _safe_diameter(
        self,
        graph: nx.Graph,
    ) -> float:
        try:
            if graph.number_of_nodes() <= 1:
                return 0.0

            return safe_float(nx.diameter(graph))
        except nx.NetworkXException:
            return 0.0

    def _safe_global_efficiency(
        self,
        graph: nx.Graph,
    ) -> float:
        try:
            return safe_float(nx.global_efficiency(graph))
        except nx.NetworkXException:
            return 0.0

    def _safe_is_planar(
        self,
        graph: nx.Graph,
    ) -> bool:
        try:
            is_planar, _embedding = nx.check_planarity(graph)
            return bool(is_planar)
        except nx.NetworkXException:
            return False

    def _core_number_values(
        self,
        graph: nx.Graph,
    ) -> np.ndarray:
        try:
            if graph.number_of_nodes() == 0:
                return np.asarray([], dtype=float)

            core_numbers = nx.core_number(graph)

            return np.asarray(
                list(core_numbers.values()),
                dtype=float,
            )
        except nx.NetworkXException:
            return np.asarray([], dtype=float)

    def _safe_degree_assortativity(
        self,
        graph: nx.Graph,
    ) -> float:
        try:
            value = nx.degree_assortativity_coefficient(graph)
            return safe_float(value, default=0.0)
        except nx.NetworkXException:
            return 0.0

    def _safe_modularity(
        self,
        graph: nx.Graph,
    ) -> float:
        """
        Approximate modularity using greedy communities.
        """
        try:
            if graph.number_of_edges() == 0:
                return 0.0

            communities = nx.algorithms.community.greedy_modularity_communities(
                graph
            )

            if not communities:
                return 0.0

            modularity = nx.algorithms.community.modularity(
                graph,
                communities,
            )

            return safe_float(modularity, default=0.0)
        except nx.NetworkXException:
            return 0.0