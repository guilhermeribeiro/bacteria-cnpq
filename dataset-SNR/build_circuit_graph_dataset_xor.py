"""Build NetworkX graphs, graph statistics, and GNN-ready tensors for XOR data.

Usage:
    python3 build_circuit_graph_dataset_xor.py
    python3 build_circuit_graph_dataset_xor.py --limit 100 --output-dir xor-dataset/artifacts_sample
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from build_circuit_graph_dataset import (
    CircuitGraphRecord,
    _format_float,
    iter_jsonl,
    to_pyg_data,
    write_statistics_csv,
)
from calculate_statistics_graph import calculate_graph_statistics


EDGE_THRESHOLD = 0.01

COMPONENT_LIBRARY: dict[str, dict[str, float]] = {
    "Vazio": {"component_type_id": 0, "ymin": 0.0, "ymax": 0.0, "Kd": 0.0, "n_real": 0.0},
    "pTac": {"component_type_id": 1, "ymin": 0.0, "ymax": 0.0, "Kd": 0.0, "n_real": 0.0},
    "pTet": {"component_type_id": 2, "ymin": 0.0, "ymax": 0.0, "Kd": 0.0, "n_real": 0.0},
    "YFP": {"component_type_id": 3, "ymin": 0.02, "ymax": 1.0, "Kd": 0.1, "n_real": 1.0},
    "AmtR": {"component_type_id": 4, "ymin": 0.06, "ymax": 3.80, "Kd": 0.07, "n_real": 1.60},
    "BetI": {"component_type_id": 5, "ymin": 0.07, "ymax": 3.80, "Kd": 0.41, "n_real": 2.40},
    "AmeR": {"component_type_id": 6, "ymin": 0.20, "ymax": 3.80, "Kd": 0.09, "n_real": 1.40},
    "PhlF": {"component_type_id": 7, "ymin": 0.01, "ymax": 3.90, "Kd": 0.03, "n_real": 4.00},
    "QacR": {"component_type_id": 8, "ymin": 0.01, "ymax": 2.40, "Kd": 0.05, "n_real": 2.70},
    "SrpR": {"component_type_id": 9, "ymin": 0.003, "ymax": 1.30, "Kd": 0.01, "n_real": 2.90},
}

MAX_COMPONENT_TYPE_ID = max(
    component["component_type_id"] for component in COMPONENT_LIBRARY.values()
)

NODE_FEATURES = [
    "is_input",
    "is_output",
    "is_empty",
    "component_type_norm",
    "confidence_pct_norm",
    "ymin",
    "ymax",
    "n_real",
    "Kd",
    "node_index_norm",
]
EDGE_FEATURES = [
    "weight",
    "weight_norm",
    "source_component_type_norm",
    "target_component_type_norm",
    "source_confidence_pct_norm",
    "target_confidence_pct_norm",
]


class XorCircuitGraphDataset(Dataset):
    """PyTorch dataset that exposes each XOR circuit as graph tensors."""

    def __init__(self, jsonl_path: str | Path, limit: int | None = None):
        self.records = [
            build_record(raw_record)
            for raw_record in iter_jsonl(Path(jsonl_path), limit=limit)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        item = dict(record.tensors)
        item["circuit_id"] = record.circuit_id
        item["statistics"] = record.statistics
        return item


def _slot_by_index(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(slot["slot"]): slot for slot in record.get("slots", [])}


def _component_params(winner: str) -> dict[str, float]:
    return COMPONENT_LIBRARY.get(winner, COMPONENT_LIBRARY["Vazio"])


def _confidence_norm(slot: dict[str, Any]) -> float:
    return float(slot.get("confidence_pct", 0.0)) / 100.0


def _component_type_norm(component_type_id: float) -> float:
    return float(component_type_id) / float(MAX_COMPONENT_TYPE_ID)


def visualize_graph(
    G: nx.DiGraph,
    statistics: dict[str, Any] | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Visualize an XOR circuit graph with slot, confidence, and tensor info."""

    if statistics is None:
        statistics = calculate_graph_statistics(G)

    pos = nx.spring_layout(G, seed=7)
    node_colors = []
    for _, attrs in G.nodes(data=True):
        if attrs.get("is_input", 0.0) == 1.0:
            node_colors.append("#77aadd")
        elif attrs.get("is_output", 0.0) == 1.0:
            node_colors.append("#ee8866")
        elif attrs.get("is_empty", 0.0) == 1.0:
            node_colors.append("#dddddd")
        else:
            node_colors.append("#99dd99")

    node_labels = {
        node_id: "\n".join(
            [
                f"{attrs.get('label', f'node_{node_id}')}: {attrs.get('protein_name', 'Unknown')}",
                f"in={_format_float(attrs.get('is_input'))} out={_format_float(attrs.get('is_output'))}",
                f"empty={_format_float(attrs.get('is_empty'))} conf={_format_float(attrs.get('confidence_pct'))}%",
                (
                    f"ymin={_format_float(attrs.get('ymin'))} "
                    f"ymax={_format_float(attrs.get('ymax'))}"
                ),
                f"Kd={_format_float(attrs.get('Kd'))} n={_format_float(attrs.get('n_real'))}",
            ]
        )
        for node_id, attrs in G.nodes(data=True)
    }

    edge_labels = {
        (source, target): f"w={_format_float(attrs.get('weight'), 4)}"
        for source, target, attrs in G.edges(data=True)
    }

    fig, (ax_graph, ax_info) = plt.subplots(
        1,
        2,
        figsize=(15, 8),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=2700,
        edgecolors="#333333",
        linewidths=1.0,
        ax=ax_graph,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#555555",
        arrows=True,
        arrowsize=18,
        width=1.6,
        connectionstyle="arc3,rad=0.08",
        ax=ax_graph,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, ax=ax_graph)
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.55,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
        ax=ax_graph,
    )

    circuit_id = G.graph.get("circuit_id", "Unknown")
    snr = G.graph.get("snr", 0.0)
    ax_graph.set_title(f"XOR Circuit {circuit_id} | SNR={_format_float(snr, 5)}")
    ax_graph.axis("off")

    info_lines = [
        "Graph metadata",
        f"circuit_id: {circuit_id}",
        f"gate: {G.graph.get('gate', 'Unknown')}",
        f"SNR: {_format_float(snr, 5)}",
        f"algorithm: {G.graph.get('algorithm_mnemonic', 'Unknown')}",
        f"edge_threshold: {EDGE_THRESHOLD}",
        "",
        "Meta-features",
        f"num_nodes: {statistics.get('num_nodes')}",
        f"num_edges: {statistics.get('num_edges')}",
        f"density: {_format_float(statistics.get('density'))}",
        f"avg_clustering: {_format_float(statistics.get('average_clustering'))}",
        f"transitivity: {_format_float(statistics.get('transitivity'))}",
        f"reciprocity: {_format_float(statistics.get('reciprocity'))}",
        f"is_DAG: {statistics.get('is_directed_acyclic_graph')}",
        f"weak_components: {statistics.get('num_weak_components')}",
        f"strong_components: {statistics.get('num_strong_components')}",
        "",
        "Node features (x)",
        ", ".join(NODE_FEATURES),
        "",
        "Edge features (edge_attr)",
        ", ".join(EDGE_FEATURES),
    ]

    if tensors is not None:
        info_lines.extend(
            [
                "",
                "Tensor shapes",
                f"x: {tuple(tensors['x'].shape)}",
                f"edge_index: {tuple(tensors['edge_index'].shape)}",
                f"edge_attr: {tuple(tensors['edge_attr'].shape)}",
                f"y: {tuple(tensors['y'].shape)} = {_format_float(tensors['y'].item(), 5)}",
            ]
        )

    ax_info.text(
        0.0,
        1.0,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )
    ax_info.axis("off")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def circuit_to_networkx(record: dict[str, Any]) -> nx.DiGraph:
    """Convert one XOR JSON object into a directed NetworkX graph.

    The XOR file stores continuous wiring weights instead of UCF integer IDs.
    To stay compatible with the existing graph builder, W[i, j] > threshold is
    treated as a regulatory edge from node j to node i.
    """

    matrix = record["wiring_matrix"]
    slots = _slot_by_index(record)
    graph = nx.DiGraph(
        circuit_id=record.get("id"),
        gate=record.get("gate"),
        snr=float(record.get("snr", 0.0)),
        algorithm_mnemonic=record.get("algorithm_mnemonic"),
        edge_threshold=EDGE_THRESHOLD,
    )

    node_count = len(matrix)
    for slot_index in range(node_count):
        node_id = slot_index + 1
        slot = slots.get(slot_index, {"slot": slot_index, "winner": "Vazio", "confidence_pct": 0.0})
        winner = str(slot.get("winner", "Vazio"))
        params = _component_params(winner)
        component_type_id = float(params["component_type_id"])
        graph.add_node(
            node_id,
            label=f"slot_{slot_index}",
            slot=slot_index,
            protein_name=winner,
            is_input=float(winner in {"pTac", "pTet"} or slot_index in {0, 1}),
            is_output=float(winner == "YFP" or slot_index == node_count - 1),
            is_empty=float(winner == "Vazio"),
            confidence_pct=float(slot.get("confidence_pct", 0.0)),
            confidence_pct_norm=_confidence_norm(slot),
            component_type_id=component_type_id,
            component_type_norm=_component_type_norm(component_type_id),
            ymin=float(params["ymin"]),
            ymax=float(params["ymax"]),
            n_real=float(params["n_real"]),
            Kd=float(params["Kd"]),
        )

    max_weight = max(
        (abs(float(weight)) for row in matrix for weight in row if abs(float(weight)) > EDGE_THRESHOLD),
        default=1.0,
    )
    for target_index, row in enumerate(matrix, start=1):
        for source_index, weight in enumerate(row, start=1):
            weight = float(weight)
            if abs(weight) <= EDGE_THRESHOLD:
                continue
            source_attrs = graph.nodes[source_index]
            target_attrs = graph.nodes[target_index]
            graph.add_edge(
                source_index,
                target_index,
                weight=weight,
                weight_norm=weight / max_weight,
                source_component_type_norm=float(source_attrs.get("component_type_norm", 0.0)),
                target_component_type_norm=float(target_attrs.get("component_type_norm", 0.0)),
                source_confidence_pct_norm=float(source_attrs.get("confidence_pct_norm", 0.0)),
                target_confidence_pct_norm=float(target_attrs.get("confidence_pct_norm", 0.0)),
            )

    return graph


def graph_to_gnn_tensors(graph: nx.DiGraph, target: float) -> dict[str, torch.Tensor]:
    nodes = sorted(graph.nodes)
    node_positions = {node_id: index for index, node_id in enumerate(nodes)}
    denominator = max(len(nodes) - 1, 1)

    x = []
    for node_id in nodes:
        attrs = graph.nodes[node_id]
        x.append(
            [
                float(attrs.get("is_input", 0.0)),
                float(attrs.get("is_output", 0.0)),
                float(attrs.get("is_empty", 0.0)),
                float(attrs.get("component_type_norm", 0.0)),
                float(attrs.get("confidence_pct_norm", 0.0)),
                float(attrs.get("ymin", 0.0)),
                float(attrs.get("ymax", 0.0)),
                float(attrs.get("n_real", 0.0)),
                float(attrs.get("Kd", 0.0)),
                float((node_id - 1) / denominator),
            ]
        )

    edge_index = []
    edge_attr = []
    for source, target_node, attrs in graph.edges(data=True):
        edge_index.append([node_positions[source], node_positions[target_node]])
        edge_attr.append(
            [
                float(attrs.get("weight", 0.0)),
                float(attrs.get("weight_norm", 0.0)),
                float(attrs.get("source_component_type_norm", 0.0)),
                float(attrs.get("target_component_type_norm", 0.0)),
                float(attrs.get("source_confidence_pct_norm", 0.0)),
                float(attrs.get("target_confidence_pct_norm", 0.0)),
            ]
        )

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, len(EDGE_FEATURES)), dtype=torch.float32)

    return {
        "x": torch.tensor(np.asarray(x), dtype=torch.float32),
        "edge_index": edge_index_tensor,
        "edge_attr": edge_attr_tensor,
        "y": torch.tensor([float(target)], dtype=torch.float32),
    }


def build_record(raw_record: dict[str, Any]) -> CircuitGraphRecord:
    graph = circuit_to_networkx(raw_record)
    statistics = calculate_graph_statistics(graph)
    target = float(raw_record["snr"])
    return CircuitGraphRecord(
        circuit_id=str(raw_record["id"]),
        graph=graph,
        statistics=statistics,
        tensors=graph_to_gnn_tensors(graph, target),
        target=target,
    )


def build_artifacts(input_path: Path, output_dir: Path, limit: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs: list[nx.DiGraph] = []
    tensor_records: list[dict[str, Any]] = []
    statistics_rows: list[dict[str, Any]] = []

    for raw_record in iter_jsonl(input_path, limit=limit):
        record = build_record(raw_record)
        graphs.append(record.graph)
        tensor_records.append(
            {
                "circuit_id": record.circuit_id,
                "x": record.tensors["x"],
                "edge_index": record.tensors["edge_index"],
                "edge_attr": record.tensors["edge_attr"],
                "y": record.tensors["y"],
            }
        )
        statistics_rows.append(
            {
                "circuit_id": record.circuit_id,
                "Y": record.target,
                "snr": record.target,
                "gate": raw_record.get("gate"),
                "algorithm_mnemonic": raw_record.get("algorithm_mnemonic"),
                **record.statistics,
            }
        )

    with (output_dir / "networkx_graphs.pkl").open("wb") as handle:
        pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    torch.save(
        {
            "node_features": NODE_FEATURES,
            "edge_features": EDGE_FEATURES,
            "records": tensor_records,
        },
        output_dir / "gnn_tensors.pt",
    )
    write_statistics_csv(output_dir / "graph_statistics.csv", statistics_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("xor-dataset/circuit_dataset_XOR.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("xor-dataset/artifacts"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_artifacts(args.input, args.output_dir, args.limit)
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
