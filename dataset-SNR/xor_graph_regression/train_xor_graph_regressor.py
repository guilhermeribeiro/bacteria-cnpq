"""Treina uma GNN para predizer SNR diretamente dos grafos de circuitos XOR.

O construtor abaixo reproduz a representacao de ``build_circuit_graph_dataset.py``:

* ``matrix_W[i][j] > 0`` cria uma aresta dirigida do no ``j`` para o no ``i``;
* os nos recebem is_input, is_output e os parametros de Hill;
* as arestas recebem o identificador e os parametros da porta repressora.

Nenhuma estatistica global (densidade, clustering, centralidade etc.) e calculada ou
fornecida ao modelo. A entrada do regressor e um batch de grafos PyG.

Uso, a partir da raiz do projeto:

    python xor_graph_regression/train_xor_graph_regressor.py
    python xor_graph_regression/train_xor_graph_regressor.py --epochs 200
    python xor_graph_regression/train_xor_graph_regressor.py --limit 100 --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterator

import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool
except ImportError as exc:
    raise SystemExit(
        "torch_geometric e necessario. Instale com: "
        "python -m pip install torch-geometric"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "xor-dataset" / "circuit_dataset_XOR.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs" / "default"

NODE_FEATURES = [
    "is_input",
    "is_output",
    "ymin",
    "ymax",
    "n_real",
    "Kd",
    "node_index_norm",
]
EDGE_FEATURES = ["repressor_id_norm", "ymin", "ymax", "Kd", "n"]

# Copia deliberada da biblioteca usada por build_circuit_graph_dataset.py. Manter
# estes valores sincronizados preserva exatamente a mesma representacao do grafo.
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


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Le os registros sem carregar o JSONL inteiro na memoria."""
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def circuit_to_networkx(record: dict[str, Any]) -> nx.DiGraph:
    """Constroi o mesmo DiGraph definido em build_circuit_graph_dataset.py."""
    matrix = record["matrix_W"]
    components = record.get("components", {})
    metadata = record.get("metadata_for_humans", {})
    graph = nx.DiGraph(
        circuit_id=record.get("id"),
        gate=record.get("gate"),
        snr=float(record["snr"]),
        ensemble_size=int(record.get("ensemble_size", 0)),
    )

    for node_id in range(1, len(matrix) + 1):
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
        for source_index, raw_repressor_id in enumerate(row, start=1):
            repressor_id = int(raw_repressor_id)
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


def networkx_to_pyg(graph: nx.DiGraph) -> Data:
    """Converte o grafo, sem meta-features, em um objeto PyG Data."""
    nodes = sorted(graph.nodes)
    node_positions = {node_id: index for index, node_id in enumerate(nodes)}
    denominator = max(len(nodes) - 1, 1)

    node_rows = []
    for node_id in nodes:
        attrs = graph.nodes[node_id]
        node_rows.append(
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

    edge_pairs = []
    edge_rows = []
    for source, target, attrs in graph.edges(data=True):
        edge_pairs.append([node_positions[source], node_positions[target]])
        edge_rows.append(
            [
                float(attrs.get("repressor_id", 0.0)) / 20.0,
                float(attrs.get("ymin", 0.0)),
                float(attrs.get("ymax", 0.0)),
                float(attrs.get("Kd", 0.0)),
                float(attrs.get("n", 0.0)),
            ]
        )

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_rows, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(EDGE_FEATURES)), dtype=torch.float32)

    data = Data(
        x=torch.tensor(node_rows, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(graph.graph["snr"])], dtype=torch.float32),
    )
    data.circuit_id = str(graph.graph["circuit_id"])
    return data


def load_graph_dataset(path: Path, limit: int | None = None) -> list[Data]:
    """Materializa cada linha do JSONL como um grafo PyG independente."""
    return [networkx_to_pyg(circuit_to_networkx(row)) for row in iter_jsonl(path, limit)]


def split_dataset(
    dataset: list[Data],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    stratify_bins: int,
    outlier_threshold: float,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Divide os grafos preservando proporcionalmente a distribuicao de SNR.

    Os valores abaixo de ``outlier_threshold`` sao ordenados e separados em
    faixas de quantis com aproximadamente o mesmo numero de amostras. Os valores
    acima do limiar formam uma faixa rara propria. Cada faixa e distribuida nas
    proporcoes de treino, validacao e teste.
    """
    if len(dataset) < 3:
        raise ValueError("Sao necessarios pelo menos tres grafos para train/val/test.")
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("train-ratio e val-ratio devem estar entre zero e um.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio deve ser menor que 1.")
    if stratify_bins < 1:
        raise ValueError("stratify-bins deve ser pelo menos 1.")

    regular_indices = [
        index
        for index, graph in enumerate(dataset)
        if float(graph.y.item()) < outlier_threshold
    ]
    outlier_indices = [
        index
        for index, graph in enumerate(dataset)
        if float(graph.y.item()) >= outlier_threshold
    ]
    regular_indices.sort(key=lambda index: float(dataset[index].y.item()))

    # A separacao por posicao equivale a bins de quantis, mas evita depender de
    # pandas/sklearn e funciona mesmo quando muitos alvos possuem o mesmo valor.
    smallest_ratio = min(train_ratio, val_ratio, 1.0 - train_ratio - val_ratio)
    minimum_bin_size = math.ceil(1.0 / smallest_ratio)
    effective_bins = min(
        stratify_bins,
        max(1, len(regular_indices) // minimum_bin_size),
    )
    strata: list[list[int]] = [[] for _ in range(effective_bins)]
    for rank, index in enumerate(regular_indices):
        bin_index = min(
            (rank * effective_bins) // max(len(regular_indices), 1),
            effective_bins - 1,
        )
        strata[bin_index].append(index)
    if outlier_indices:
        strata.append(outlier_indices)

    rng = random.Random(seed)
    split_indices: list[list[int]] = [[], [], []]
    test_ratio = 1.0 - train_ratio - val_ratio

    for stratum in strata:
        if not stratum:
            continue
        rng.shuffle(stratum)
        size = len(stratum)

        # Largest remainder: arredonda as quantidades mantendo o total da faixa.
        exact_counts = [
            size * train_ratio,
            size * val_ratio,
            size * test_ratio,
        ]
        counts = [math.floor(value) for value in exact_counts]
        remaining = size - sum(counts)
        remainder_order = sorted(
            range(3),
            key=lambda position: exact_counts[position] - counts[position],
            reverse=True,
        )
        for position in remainder_order[:remaining]:
            counts[position] += 1

        # Uma faixa rara com pelo menos tres exemplos deve aparecer nos tres
        # conjuntos. A compensacao e retirada do maior conjunto da faixa.
        if size >= 3:
            for position in range(3):
                if counts[position] == 0:
                    donor = max(range(3), key=lambda candidate: counts[candidate])
                    counts[donor] -= 1
                    counts[position] += 1

        train_end = counts[0]
        val_end = train_end + counts[1]
        split_indices[0].extend(stratum[:train_end])
        split_indices[1].extend(stratum[train_end:val_end])
        split_indices[2].extend(stratum[val_end:])

    # O arredondamento dentro de cada faixa pode deixar a contagem global um ou
    # dois elementos distante da proporcao pedida. Reequilibramos apenas casos
    # regulares, preservando a distribuicao 6/1/1 (ou equivalente) dos raros.
    global_exact_counts = [
        len(dataset) * train_ratio,
        len(dataset) * val_ratio,
        len(dataset) * test_ratio,
    ]
    desired_counts = [math.floor(value) for value in global_exact_counts]
    global_remaining = len(dataset) - sum(desired_counts)
    global_remainder_order = sorted(
        range(3),
        key=lambda position: global_exact_counts[position] - desired_counts[position],
        reverse=True,
    )
    for position in global_remainder_order[:global_remaining]:
        desired_counts[position] += 1

    while [len(indices) for indices in split_indices] != desired_counts:
        receiver = next(
            position
            for position in range(3)
            if len(split_indices[position]) < desired_counts[position]
        )
        donor = next(
            position
            for position in range(3)
            if len(split_indices[position]) > desired_counts[position]
        )
        donor_indices = split_indices[donor]
        movable_position = next(
            (
                position
                for position, index in enumerate(donor_indices)
                if float(dataset[index].y.item()) < outlier_threshold
            ),
            0,
        )
        split_indices[receiver].append(donor_indices.pop(movable_position))

    for indices in split_indices:
        rng.shuffle(indices)
    if any(not indices for indices in split_indices):
        raise ValueError(
            "O split estratificado produziu um conjunto vazio; use mais dados "
            "ou menos stratify-bins."
        )

    return tuple(
        [dataset[index] for index in indices] for indices in split_indices
    )


def describe_split(name: str, graphs: list[Data], outlier_threshold: float) -> str:
    """Resume a distribuicao do alvo para tornar o balanceamento auditavel."""
    targets = torch.tensor([float(graph.y.item()) for graph in graphs])
    outliers = int((targets >= outlier_threshold).sum().item())
    return (
        f"{name}={len(graphs)} "
        f"SNR[min={targets.min().item():.4f}, media={targets.mean().item():.4f}, "
        f"max={targets.max().item():.4f}, >=limiar={outliers}]"
    )


def fit_and_apply_standardization(
    train_data: list[Data], all_data: list[Data]
) -> dict[str, torch.Tensor]:
    """Ajusta a normalizacao apenas no treino, evitando vazamento de dados."""
    node_values = torch.cat([graph.x for graph in train_data], dim=0)
    node_mean = node_values.mean(dim=0)
    node_std = node_values.std(dim=0, unbiased=False).clamp_min(1e-6)

    train_edges = [graph.edge_attr for graph in train_data if graph.edge_attr.numel()]
    if train_edges:
        edge_values = torch.cat(train_edges, dim=0)
        edge_mean = edge_values.mean(dim=0)
        edge_std = edge_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    else:
        edge_mean = torch.zeros(len(EDGE_FEATURES))
        edge_std = torch.ones(len(EDGE_FEATURES))

    target_values = torch.cat([graph.y for graph in train_data])
    target_mean = target_values.mean()
    target_std = target_values.std(unbiased=False).clamp_min(1e-6)

    for graph in all_data:
        graph.x = (graph.x - node_mean) / node_std
        if graph.edge_attr.numel():
            graph.edge_attr = (graph.edge_attr - edge_mean) / edge_std
        graph.y_scaled = (graph.y - target_mean) / target_std

    return {
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


class SNRGraphRegressor(nn.Module):
    """Regressor de grafo que usa atributos de nos e arestas nas convolucoes."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.node_encoder = nn.Linear(len(NODE_FEATURES), hidden_channels)
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        for _ in range(num_layers):
            update_network = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convolutions.append(
                GINEConv(update_network, edge_dim=len(EDGE_FEATURES), train_eps=True)
            )
            self.normalizations.append(nn.LayerNorm(hidden_channels))

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, batch: Data) -> torch.Tensor:
        x = self.node_encoder(batch.x)
        for convolution, normalization in zip(
            self.convolutions, self.normalizations
        ):
            residual = x
            x = convolution(x, batch.edge_index, batch.edge_attr)
            x = normalization(F.relu(x))
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
        graph_embedding = global_add_pool(x, batch.batch)
        return self.readout(graph_embedding).view(-1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Executa uma epoca usando RMSE no alvo padronizado como loss."""
    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(training):
            prediction = model(batch)
            mse = F.mse_loss(prediction, batch.y_scaled.view(-1))
            # O epsilon evita uma derivada indefinida de sqrt exatamente em zero.
            loss = torch.sqrt(mse + 1e-8)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_squared_error += float(mse.item()) * batch.num_graphs
        total_graphs += int(batch.num_graphs)
    return math.sqrt(total_squared_error / max(total_graphs, 1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    rows: list[dict[str, float | str]] = []

    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch).cpu() * target_std + target_mean
        target = batch.y.view(-1).cpu()
        predictions.append(prediction)
        targets.append(target)
        circuit_ids = getattr(batch, "circuit_id", [""] * batch.num_graphs)
        for circuit_id, observed, predicted in zip(circuit_ids, target, prediction):
            observed_value = float(observed.item())
            predicted_value = float(predicted.item())
            rows.append(
                {
                    "circuit_id": str(circuit_id),
                    "snr_true": observed_value,
                    "snr_pred": predicted_value,
                    "absolute_error": abs(observed_value - predicted_value),
                }
            )

    y_pred = torch.cat(predictions)
    y_true = torch.cat(targets)
    mse = float(F.mse_loss(y_pred, y_true).item())
    mae = float(F.l1_loss(y_pred, y_true).item())
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = float("nan") if ss_tot <= 1e-12 else 1.0 - float((ss_res / ss_tot).item())
    return {"mae": mae, "mse": mse, "rmse": math.sqrt(mse), "r2": r2}, rows


def write_predictions(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["circuit_id", "snr_true", "snr_pred", "absolute_error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predicao de SNR diretamente de grafos de circuitos XOR."
    )
    # parser.add_argument("--input", type=Path, default='/Users/guilhermeribeiro/MEGA/PyCharmProjects/bacteria-cnpq/dataset-SNR/xor-dataset/circuit_dataset_XOR.jsonl')
    parser.add_argument("--input", type=Path,
                        default=r'C:\Users\guilh\PycharmProjects\bacteria-cnpq\dataset-SNR\xor-dataset\circuit_dataset_XOR.jsonl')
    # parser.add_argument("--output-dir", type=Path,
    #                     default='/Users/guilhermeribeiro/MEGA/PyCharmProjects/bacteria-cnpq/dataset-SNR/xor_graph_regression'
    #                             '/output_global_add_pool')
    parser.add_argument("--output-dir", type=Path,
                        default=r'C:\Users\guilh\PycharmProjects\bacteria-cnpq\dataset-SNR\xor_graph_regression\output_global_add_pool_seed_ajusted')
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--stratify-bins",
        type=int,
        default=10,
        help="Numero de faixas de quantis usadas no split estratificado.",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=2.0,
        help="SNR a partir do qual uma instancia entra na faixa rara propria.",
    )
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="auto, cpu, cuda ou mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()


    if not args.input.is_file():
        raise FileNotFoundError(f"JSONL nao encontrado: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(30):
        random.seed(i)
        torch.manual_seed(i)

        dataset = load_graph_dataset(args.input, args.limit)
        train_data, val_data, test_data = split_dataset(
            dataset,
            args.train_ratio,
            args.val_ratio,
            i, #args.seed,
            args.stratify_bins,
            args.outlier_threshold,
        )
        preprocessing = fit_and_apply_standardization(train_data, dataset)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)
        device = choose_device(args.device)

        model = SNRGraphRegressor(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        checkpoint_path = args.output_dir / f"best_model_{i}.pt"
        history: list[dict[str, float | int]] = []

        print(
            f"Grafos: {len(dataset)} "
            f"(treino={len(train_data)}, validacao={len(val_data)}, teste={len(test_data)})"
        )
        print(describe_split("treino", train_data, args.outlier_threshold))
        print(describe_split("validacao", val_data, args.outlier_threshold))
        print(describe_split("teste", test_data, args.outlier_threshold))
        print(f"Dispositivo: {device}")

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            val_loss = run_epoch(model, val_loader, device)

            train_metrics, _ = evaluate(
                model,
                train_loader,
                device,
                preprocessing["target_mean"],
                preprocessing["target_std"],
            )
            val_metrics, _ = evaluate(
                model,
                val_loader,
                device,
                preprocessing["target_mean"],
                preprocessing["target_std"],
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_scaled_rmse_loss": train_loss,
                    "val_scaled_rmse_loss": val_loss,
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "train_mse": train_metrics["mse"],
                    "train_r2": train_metrics["r2"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mse": val_metrics["mse"],
                    "val_r2": val_metrics["r2"],
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "node_features": NODE_FEATURES,
                        "edge_features": EDGE_FEATURES,
                        "preprocessing": preprocessing,
                        "config": {
                            "hidden_channels": args.hidden_channels,
                            "num_layers": args.num_layers,
                            "dropout": args.dropout,
                            "seed": args.seed,
                        },
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            print(
                f"epoca={epoch:03d} "
                f"loss_rmse_escalado(treino={train_loss:.6f}, val={val_loss:.6f}) | "
                f"treino: MAE={train_metrics['mae']:.6f} "
                f"RMSE={train_metrics['rmse']:.6f} "
                f"MSE={train_metrics['mse']:.6f} R2={train_metrics['r2']:.6f} | "
                f"validacao: MAE={val_metrics['mae']:.6f} "
                f"RMSE={val_metrics['rmse']:.6f} "
                f"MSE={val_metrics['mse']:.6f} R2={val_metrics['r2']:.6f}"
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping na epoca {epoch}.")
                break

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics, predictions = evaluate(
            model,
            test_loader,
            device,
            preprocessing["target_mean"],
            preprocessing["target_std"],
        )

        write_predictions(args.output_dir / f"test_predictions_{i}.csv", predictions)
        with (args.output_dir / f"metrics_{i}.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=True)
        with (args.output_dir / f"history_{i}.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        print(
            "Teste: "
            f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
            f"MSE={metrics['mse']:.6f} R2={metrics['r2']:.6f}"
        )
        print(f"Modelo salvo em: {checkpoint_path}")
        print(f"Predicoes salvas em: {args.output_dir / f'test_predictions_{i}.csv'}")


if __name__ == "__main__":
    main()
