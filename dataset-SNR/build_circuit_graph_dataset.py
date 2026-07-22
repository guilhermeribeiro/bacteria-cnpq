"""Build NetworkX graphs, graph statistics, and GNN-ready tensors.

Usage:
    python3 build_circuit_graph_dataset.py
    python3 build_circuit_graph_dataset.py --limit 100 --output-dir artifacts_sample
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

from calculate_statistics_graph import calculate_graph_statistics


UCF_GATE_LIBRARY: dict[int, dict[str, float | str]] = {
    1: {"protein_name": "AmtR", "ucf_gate_name": "A1_AmtR", "ymin": 0.06, "ymax": 3.80, "Kd": 0.07, "n": 1.60},
    2: {"protein_name": "BM3R1", "ucf_gate_name": "B1_BM3R1", "ymin": 0.004, "ymax": 0.50, "Kd": 0.04, "n": 3.40},
    3: {"protein_name": "BM3R1", "ucf_gate_name": "B2_BM3R1", "ymin": 0.005, "ymax": 0.50, "Kd": 0.15, "n": 2.90},
    4: {"protein_name": "BM3R1", "ucf_gate_name": "B3_BM3R1", "ymin": 0.01, "ymax": 0.80, "Kd": 0.26, "n": 3.40},
    5: {"protein_name": "BetI", "ucf_gate_name": "E1_BetI", "ymin": 0.07, "ymax": 3.80, "Kd": 0.41, "n": 2.40},
    6: {"protein_name": "AmeR", "ucf_gate_name": "F1_AmeR", "ymin": 0.20, "ymax": 3.80, "Kd": 0.09, "n": 1.40},
    7: {"protein_name": "HlyIIR", "ucf_gate_name": "H1_HlyIIR", "ymin": 0.07, "ymax": 2.50, "Kd": 0.19, "n": 2.60},
    8: {"protein_name": "IcaRA", "ucf_gate_name": "I1_IcaRA", "ymin": 0.08, "ymax": 2.20, "Kd": 0.10, "n": 1.40},
    9: {"protein_name": "LitR", "ucf_gate_name": "L1_LitR", "ymin": 0.07, "ymax": 4.30, "Kd": 0.05, "n": 1.70},
    10: {"protein_name": "LmrA", "ucf_gate_name": "N1_LmrA", "ymin": 0.20, "ymax": 2.20, "Kd": 0.18, "n": 2.10},
    11: {"protein_name": "PhlF", "ucf_gate_name": "P1_PhlF", "ymin": 0.01, "ymax": 3.90, "Kd": 0.03, "n": 4.00},
    12: {"protein_name": "PhlF", "ucf_gate_name": "P2_PhlF", "ymin": 0.02, "ymax": 4.10, "Kd": 0.13, "n": 3.90},
    13: {"protein_name": "PhlF", "ucf_gate_name": "P3_PhlF", "ymin": 0.02, "ymax": 6.80, "Kd": 0.23, "n": 4.20},
    14: {"protein_name": "QacR", "ucf_gate_name": "Q1_QacR", "ymin": 0.01, "ymax": 2.40, "Kd": 0.05, "n": 2.70},
    15: {"protein_name": "QacR", "ucf_gate_name": "Q2_QacR", "ymin": 0.03, "ymax": 2.80, "Kd": 0.21, "n": 2.40},
    16: {"protein_name": "PsrA", "ucf_gate_name": "R1_PsrA", "ymin": 0.20, "ymax": 5.90, "Kd": 0.19, "n": 1.80},
    17: {"protein_name": "SrpR", "ucf_gate_name": "S1_SrpR", "ymin": 0.003, "ymax": 1.30, "Kd": 0.01, "n": 2.90},
    18: {"protein_name": "SrpR", "ucf_gate_name": "S2_SrpR", "ymin": 0.003, "ymax": 2.10, "Kd": 0.04, "n": 2.60},
    19: {"protein_name": "SrpR", "ucf_gate_name": "S3_SrpR", "ymin": 0.004, "ymax": 2.10, "Kd": 0.06, "n": 2.80},
    20: {"protein_name": "SrpR", "ucf_gate_name": "S4_SrpR", "ymin": 0.007, "ymax": 2.10, "Kd": 0.10, "n": 2.80},
}

NODE_FEATURES = ["is_input", "is_output", "ymin", "ymax", "n_real", "Kd", "node_index_norm"]
EDGE_FEATURES = ["repressor_id_norm", "ymin", "ymax", "Kd", "n"]


@dataclass
class CircuitGraphRecord:
    circuit_id: str
    graph: nx.DiGraph
    statistics: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    target: float


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def visualize_graph(
    G: nx.DiGraph,
    statistics: dict[str, Any] | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Visualize a circuit graph with biological and ML-ready information.

    The graph drawing shows:
    - node attributes used in ``x``: input/output flags and Hill parameters;
    - edge attributes used in ``edge_attr``: repressor id and UCF parameters;
    - global graph metadata/statistics and tensor shapes in a side panel.
    """

    if statistics is None:
        statistics = calculate_graph_statistics(G)

    pos = nx.spring_layout(G, seed=7)
    node_colors = []
    for _, attrs in G.nodes(data=True):
        if attrs.get("is_input", 0.0) == 1.0:
            node_colors.append("#77aadd")
        elif attrs.get("is_output", 0.0) == 1.0:
            node_colors.append("#ee8866")
        else:
            node_colors.append("#99dd99")

    node_labels = {
        node_id: "\n".join(
            [
                f"{attrs.get('label', f'node_{node_id}')}: {attrs.get('protein_name', 'Unknown')}",
                f"in={_format_float(attrs.get('is_input'))} out={_format_float(attrs.get('is_output'))}",
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
        (source, target): "\n".join(
            [
                f"{attrs.get('repressor_id', '?')}: {attrs.get('repressor_name', 'Unknown')}",
                f"Kd={_format_float(attrs.get('Kd'))} n={_format_float(attrs.get('n'))}",
            ]
        )
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
    ax_graph.set_title(f"Circuit {circuit_id} | Y/SNR={_format_float(snr, 5)}")
    ax_graph.axis("off")

    info_lines = [
        "Graph metadata",
        f"circuit_id: {circuit_id}",
        f"gate: {G.graph.get('gate', 'Unknown')}",
        f"Y/SNR: {_format_float(snr, 5)}",
        f"ensemble_size: {G.graph.get('ensemble_size', 'Unknown')}",
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
    """Convert one circuit JSON object into a directed NetworkX graph.

    README_Dataset.md defines W[i, j] > 0 as a regulatory edge from node j to
    node i. Node ids are kept 1-based to match node_1, node_2, ...
    """

    matrix = record["matrix_W"]
    components = record.get("components", {})
    metadata = record.get("metadata_for_humans", {})
    graph = nx.DiGraph(
        circuit_id=record.get("id"),
        gate=record.get("gate"),
        snr=float(record.get("snr", 0.0)),
        ensemble_size=int(record.get("ensemble_size", 0)),
    )

    node_count = len(matrix)
    for node_id in range(1, node_count + 1):
        node_key = f"node_{node_id}"
        component = components.get(node_key, {})
        graph.add_node(
            node_id,
            label=node_key,
            protein_name=metadata.get(node_key, "Unknown"),
            is_input=float(component.get("is_input", 0.0)),
            is_output=float(component.get("is_output", 0.0)),
            ymin=float(component.get("ymin", 0.0)),
            ymax=float(component.get("ymax", 0.0)),
            n_real=float(component.get("n_real", 0.0)),
            Kd=float(component.get("Kd", 0.0)),
        )

    for target_index, row in enumerate(matrix, start=1):
        for source_index, repressor_id in enumerate(row, start=1):
            repressor_id = int(repressor_id)
            if repressor_id <= 0:
                continue
            repressor = UCF_GATE_LIBRARY.get(repressor_id, {})
            graph.add_edge(
                source_index,
                target_index,
                repressor_id=repressor_id,
                repressor_name=repressor.get("protein_name", "Unknown"),
                ucf_gate_name=repressor.get("ucf_gate_name", "Unknown"),
                ymin=float(repressor.get("ymin", 0.0)),
                ymax=float(repressor.get("ymax", 0.0)),
                Kd=float(repressor.get("Kd", 0.0)),
                n=float(repressor.get("n", 0.0)),
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
                float(attrs.get("repressor_id", 0)) / 20.0,
                float(attrs.get("ymin", 0.0)),
                float(attrs.get("ymax", 0.0)),
                float(attrs.get("Kd", 0.0)),
                float(attrs.get("n", 0.0)),
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


def to_pyg_data(tensors: dict[str, torch.Tensor]):
    """Create a torch_geometric.data.Data object when PyG is installed."""

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is not installed. Install PyTorch Geometric to use "
            "to_pyg_data(), or consume the returned PyTorch tensors directly."
        ) from exc

    return Data(
        x=tensors["x"],
        edge_index=tensors["edge_index"],
        edge_attr=tensors["edge_attr"],
        y=tensors["y"],
    )


class CircuitGraphDataset(Dataset):
    """PyTorch dataset that exposes each circuit as graph tensors."""

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


def build_record(raw_record: dict[str, Any]) -> CircuitGraphRecord:
    graph = circuit_to_networkx(raw_record)
    # visualize_graph(graph)
    statistics = calculate_graph_statistics(graph)
    target = float(raw_record["snr"])
    return CircuitGraphRecord(
        circuit_id=str(raw_record["id"]),
        graph=graph,
        statistics=statistics,
        tensors=graph_to_gnn_tensors(graph, target),
        target=target,
    )


def write_statistics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--input", type=Path, default=Path("circuit_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_artifacts(args.input, args.output_dir, args.limit)
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
