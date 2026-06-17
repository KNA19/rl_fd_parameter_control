from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np


GraphGenerator = Callable[[int, int], nx.Graph]


def clean_graph(graph: nx.Graph) -> nx.Graph:
    """
    Clean generated graph.

    Operations:
        1. Convert to simple undirected graph.
        2. Remove self-loops.
        3. Keep largest connected component if disconnected.
        4. Relabel nodes to 0, 1, ..., n-1.

    We keep connected graphs because force-directed layout evaluation and
    graph-level features are more stable for connected graphs.
    """
    simple_graph = nx.Graph(graph)
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))

    if simple_graph.number_of_nodes() == 0:
        return nx.path_graph(2)

    if not nx.is_connected(simple_graph):
        largest_component = max(nx.connected_components(simple_graph), key=len)
        simple_graph = simple_graph.subgraph(largest_component).copy()

    if simple_graph.number_of_nodes() < 2:
        return nx.path_graph(2)

    relabeled = nx.convert_node_labels_to_integers(
        simple_graph,
        first_label=0,
        ordering="default",
    )

    return nx.Graph(relabeled)


def generate_prufer_tree(
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a random tree using a Prüfer sequence.

    This avoids depending on NetworkX version-specific random_tree functions.
    """
    if n <= 1:
        return nx.path_graph(2)

    rng = np.random.default_rng(seed)

    if n == 2:
        return nx.path_graph(2)

    prufer = rng.integers(
        low=0,
        high=n,
        size=n - 2,
    ).tolist()

    degree = [1] * n

    for node in prufer:
        degree[int(node)] += 1

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for node in prufer:
        node_int = int(node)

        leaf = min(index for index in range(n) if degree[index] == 1)

        graph.add_edge(leaf, node_int)

        degree[leaf] -= 1
        degree[node_int] -= 1

    remaining = [index for index in range(n) if degree[index] == 1]

    if len(remaining) == 2:
        graph.add_edge(remaining[0], remaining[1])

    return clean_graph(graph)


def generate_erdos_renyi(
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a connected Erdős–Rényi graph.

    The edge probability is selected to produce sparse to moderately sparse
    graphs.
    """
    rng = np.random.default_rng(seed)

    fallback_graph = nx.path_graph(max(2, n))

    for attempt in range(20):
        average_degree = float(rng.uniform(2.2, 5.0))
        p = min(0.8, max(0.05, average_degree / max(1, n - 1)))

        graph = nx.gnp_random_graph(
            n=n,
            p=p,
            seed=seed + attempt,
        )

        fallback_graph = graph
        cleaned = clean_graph(graph)

        if cleaned.number_of_nodes() >= max(2, int(0.80 * n)):
            return cleaned

    return clean_graph(fallback_graph)


def generate_barabasi_albert(
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a Barabási–Albert preferential attachment graph.
    """
    rng = np.random.default_rng(seed)

    n_safe = max(2, int(n))
    max_m = max(1, min(5, n_safe // 4))
    m = int(rng.integers(1, max_m + 1))

    # NetworkX requires m < n.
    m = min(m, n_safe - 1)

    graph = nx.barabasi_albert_graph(
        n=n_safe,
        m=m,
        seed=seed,
    )

    return clean_graph(graph)


def generate_watts_strogatz(
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a Watts–Strogatz small-world graph.
    """
    rng = np.random.default_rng(seed)

    n_safe = max(2, int(n))
    max_k = min(10, n_safe - 1)

    if max_k < 2:
        return nx.path_graph(n_safe)

    possible_k_values = [
        value for value in range(2, max_k + 1) if value % 2 == 0
    ]

    if not possible_k_values:
        k = 2
    else:
        k = int(rng.choice(possible_k_values))

    p = float(rng.uniform(0.05, 0.30))

    fallback_graph = nx.path_graph(n_safe)

    for attempt in range(20):
        graph = nx.watts_strogatz_graph(
            n=n_safe,
            k=k,
            p=p,
            seed=seed + attempt,
        )

        fallback_graph = graph
        cleaned = clean_graph(graph)

        if cleaned.number_of_nodes() >= max(2, int(0.80 * n_safe)):
            return cleaned

    return clean_graph(fallback_graph)


def generate_grid(
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a 2D grid-like graph with approximately n nodes.
    """
    _ = seed

    n_safe = max(2, int(n))

    rows = max(2, int(math.sqrt(n_safe)))
    cols = max(2, int(math.ceil(n_safe / rows)))

    grid = nx.grid_2d_graph(rows, cols)

    selected_nodes = list(grid.nodes())[:n_safe]
    graph = grid.subgraph(selected_nodes).copy()

    graph = nx.convert_node_labels_to_integers(
        graph,
        first_label=0,
        ordering="default",
    )

    return clean_graph(graph)


def generate_bipartite(
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a random bipartite graph.
    """
    rng = np.random.default_rng(seed)

    n_safe = max(2, int(n))
    n_left = max(1, int(round(0.45 * n_safe)))
    n_right = max(1, n_safe - n_left)

    p = float(rng.uniform(0.08, 0.25))

    fallback_graph = nx.path_graph(n_safe)

    for attempt in range(20):
        graph = nx.bipartite.random_graph(
            n=n_left,
            m=n_right,
            p=p,
            seed=seed + attempt,
        )

        fallback_graph = graph
        cleaned = clean_graph(graph)

        if cleaned.number_of_nodes() >= max(2, int(0.70 * n_safe)):
            return cleaned

    return clean_graph(fallback_graph)


def generate_random_geometric(
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a random geometric graph.
    """
    rng = np.random.default_rng(seed)

    n_safe = max(2, int(n))
    base_radius = math.sqrt(
        max(1.0, math.log(max(3, n_safe))) / max(3, n_safe)
    )

    fallback_graph = nx.path_graph(n_safe)

    for attempt in range(20):
        radius = float(base_radius * rng.uniform(1.2, 2.2))

        graph = nx.random_geometric_graph(
            n=n_safe,
            radius=radius,
            seed=seed + attempt,
        )

        fallback_graph = graph
        cleaned = clean_graph(graph)

        if cleaned.number_of_nodes() >= max(2, int(0.70 * n_safe)):
            return cleaned

    return clean_graph(fallback_graph)


def generate_clustered(
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a simple clustered/community graph using a stochastic block model.
    """
    rng = np.random.default_rng(seed)

    n_safe = max(4, int(n))

    max_blocks = min(5, max(3, n_safe // 8))
    num_blocks = int(rng.integers(2, max_blocks + 1))

    base_size = n_safe // num_blocks
    sizes = [base_size] * num_blocks

    remaining = n_safe - sum(sizes)

    for index in range(remaining):
        sizes[index % num_blocks] += 1

    p_in = float(rng.uniform(0.25, 0.55))
    p_out = float(rng.uniform(0.02, 0.08))

    probabilities: List[List[float]] = []

    for i in range(num_blocks):
        row = []

        for j in range(num_blocks):
            row.append(p_in if i == j else p_out)

        probabilities.append(row)

    fallback_graph = nx.path_graph(n_safe)

    for attempt in range(20):
        graph = nx.stochastic_block_model(
            sizes=sizes,
            p=probabilities,
            seed=seed + attempt,
        )

        fallback_graph = graph
        cleaned = clean_graph(graph)

        if cleaned.number_of_nodes() >= max(2, int(0.70 * n_safe)):
            return cleaned

    return clean_graph(fallback_graph)


GRAPH_GENERATORS: Dict[str, GraphGenerator] = {
    "erdos_renyi": generate_erdos_renyi,
    "barabasi_albert": generate_barabasi_albert,
    "watts_strogatz": generate_watts_strogatz,
    "tree": generate_prufer_tree,
    "grid": generate_grid,
    "bipartite": generate_bipartite,
    "random_geometric": generate_random_geometric,
    "clustered": generate_clustered,
}


def generate_graph(
    family: str,
    n: int,
    seed: int,
) -> nx.Graph:
    """
    Generate a graph from a named family.
    """
    if family not in GRAPH_GENERATORS:
        available = ", ".join(sorted(GRAPH_GENERATORS.keys()))
        raise ValueError(
            f"Unknown graph family: {family}. "
            f"Available families: {available}"
        )

    graph = GRAPH_GENERATORS[family](
        int(n),
        int(seed),
    )

    graph.graph["family"] = family
    graph.graph["seed"] = int(seed)

    return clean_graph(graph)


def available_graph_families() -> Tuple[str, ...]:
    """
    Return all available graph family names.
    """
    return tuple(sorted(GRAPH_GENERATORS.keys()))