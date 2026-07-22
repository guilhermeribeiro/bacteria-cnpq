"""Graph meta-feature extraction utilities for circuit topologies."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean, pstdev
from typing import Any

import networkx as nx


def _safe_numeric_summary(values: Iterable[float], prefix: str) -> dict[str, float]:
    values = list(values)
    if not values:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
        }

    return {
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
        f"{prefix}_mean": float(mean(values)),
        f"{prefix}_std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def calculate_graph_statistics(graph: nx.DiGraph) -> dict[str, Any]:
    """Calculate structural meta-features for a directed NetworkX graph.

    The returned values are plain Python scalars so they can be serialized to
    CSV/JSONL and reused as global graph features in downstream ML pipelines.
    """

    if not isinstance(graph, nx.DiGraph):
        raise TypeError("calculate_graph_statistics expects a networkx.DiGraph")

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    undirected = graph.to_undirected()

    stats: dict[str, Any] = {
        "num_nodes": int(node_count),
        "num_edges": int(edge_count),
        "density": float(nx.density(graph)) if node_count > 1 else 0.0,
        "num_self_loops": int(nx.number_of_selfloops(graph)),
        "num_isolates": int(nx.number_of_isolates(graph)),
        "is_directed_acyclic_graph": bool(nx.is_directed_acyclic_graph(graph)),
        "reciprocity": float(nx.reciprocity(graph) or 0.0) if edge_count else 0.0,
        "transitivity": float(nx.transitivity(undirected)) if node_count > 2 else 0.0,
        "average_clustering": float(nx.average_clustering(undirected))
        if node_count > 1
        else 0.0,
    }

    stats.update(
        _safe_numeric_summary((degree for _, degree in graph.in_degree()), "in_degree")
    )
    stats.update(
        _safe_numeric_summary((degree for _, degree in graph.out_degree()), "out_degree")
    )
    stats.update(
        _safe_numeric_summary((degree for _, degree in graph.degree()), "total_degree")
    )

    weak_components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))
    stats.update(
        {
            "num_weak_components": int(len(weak_components)),
            "largest_weak_component_size": int(max(map(len, weak_components), default=0)),
            "num_strong_components": int(len(strong_components)),
            "largest_strong_component_size": int(
                max(map(len, strong_components), default=0)
            ),
        }
    )

    if node_count and undirected.number_of_edges():
        largest_component_nodes = max(nx.connected_components(undirected), key=len)
        largest_component = undirected.subgraph(largest_component_nodes)
        stats["largest_component_avg_shortest_path"] = (
            float(nx.average_shortest_path_length(largest_component))
            if largest_component.number_of_nodes() > 1
            else 0.0
        )
        stats["largest_component_diameter"] = (
            int(nx.diameter(largest_component))
            if largest_component.number_of_nodes() > 1
            else 0
        )
    else:
        stats["largest_component_avg_shortest_path"] = 0.0
        stats["largest_component_diameter"] = 0

    return stats
