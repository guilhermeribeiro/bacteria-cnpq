"""Train a Graph Neural Network to predict SNR from circuit graphs.

This script reads ``circuit_dataset.jsonl``, converts each circuit to PyG
``Data`` objects using the graph construction code in build_circuit_graph_dataset.py,
and trains a graph-level regression model with SNR as the target.

Example:
    python3 train_gnn_snr.py --epochs 100 --batch-size 128
    python3 train_gnn_snr.py --limit 1000 --epochs 20 --output-dir runs/debug_gnn
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from build_circuit_graph_dataset import build_record, iter_jsonl

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_mean_pool
except ImportError as exc:
    raise SystemExit(
        "torch_geometric is required for this script.\n"
        "Install it with: python3 -m pip install torch-geometric"
    ) from exc


class SNRGNN(nn.Module):
    """Graph-level regressor that uses node and edge features."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.dropout = dropout
        self.node_encoder = nn.Linear(node_features, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(
                GINEConv(mlp, edge_dim=edge_features, train_eps=True)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_encoder(data.x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, data.edge_attr)
            x = norm(F.relu(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        graph_embedding = global_mean_pool(x, data.batch)
        return self.regressor(graph_embedding).view(-1)


def make_pyg_dataset(jsonl_path: Path, limit: int | None = None) -> list[Data]:
    dataset = []
    for raw_record in iter_jsonl(jsonl_path, limit=limit):
        record = build_record(raw_record)
        tensors = record.tensors
        data = Data(
            x=tensors["x"],
            edge_index=tensors["edge_index"],
            edge_attr=tensors["edge_attr"],
            y=tensors["y"].view(1),
        )
        data.circuit_id = record.circuit_id
        dataset.append(data)
    return dataset


def split_dataset(
    dataset: list[Data],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Data], list[Data], list[Data]]:
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    train = [dataset[index] for index in indices[:train_end]]
    val = [dataset[index] for index in indices[train_end:val_end]]
    test = [dataset[index] for index in indices[val_end:]]
    return train, val, test


def standardize_features(train_data: list[Data], all_data: list[Data]) -> None:
    node_values = torch.cat([data.x for data in train_data], dim=0)
    node_mean = node_values.mean(dim=0)
    node_std = node_values.std(dim=0).clamp_min(1e-6)

    edge_tensors = [data.edge_attr for data in train_data if data.edge_attr.numel()]
    if edge_tensors:
        edge_values = torch.cat(edge_tensors, dim=0)
        edge_mean = edge_values.mean(dim=0)
        edge_std = edge_values.std(dim=0).clamp_min(1e-6)
    else:
        edge_mean = torch.zeros(all_data[0].edge_attr.shape[1])
        edge_std = torch.ones(all_data[0].edge_attr.shape[1])

    for data in all_data:
        data.x = (data.x - node_mean) / node_std
        if data.edge_attr.numel():
            data.edge_attr = (data.edge_attr - edge_mean) / edge_std


def transform_targets(data: Iterable[Data], use_log1p: bool) -> None:
    if not use_log1p:
        return
    for graph in data:
        graph.y = torch.log1p(graph.y)


def inverse_target(values: torch.Tensor, use_log1p: bool) -> torch.Tensor:
    return torch.expm1(values) if use_log1p else values


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch)
        target = batch.y.view(-1)
        loss = F.mse_loss(prediction, target)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += int(batch.num_graphs)

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_log1p_target: bool,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    predictions = []
    targets = []
    rows = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).detach().cpu()
        target = batch.y.view(-1).detach().cpu()
        pred_snr = inverse_target(pred, use_log1p_target)
        target_snr = inverse_target(target, use_log1p_target)

        predictions.append(pred_snr)
        targets.append(target_snr)

        circuit_ids = getattr(batch, "circuit_id", [""] * batch.num_graphs)
        for circuit_id, y_true, y_pred in zip(circuit_ids, target_snr, pred_snr):
            rows.append(
                {
                    "circuit_id": str(circuit_id),
                    "snr_true": float(y_true.item()),
                    "snr_pred": float(y_pred.item()),
                    "absolute_error": abs(float(y_true.item()) - float(y_pred.item())),
                }
            )

    y_pred = torch.cat(predictions)
    y_true = torch.cat(targets)
    mse = F.mse_loss(y_pred, y_true).item()
    mae = F.l1_loss(y_pred, y_true).item()
    rmse = math.sqrt(mse)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).clamp_min(1e-12)
    r2 = 1.0 - float((ss_res / ss_tot).item())

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, rows


def write_predictions(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["circuit_id", "snr_true", "snr_pred", "absolute_error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("circuit_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gnn_snr"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--log1p-target",
        action="store_true",
        help="Train on log1p(SNR), while reporting metrics in original SNR scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_pyg_dataset(args.input, limit=args.limit)
    if not dataset:
        raise ValueError(f"No graphs were loaded from {args.input}")

    train_data, val_data, test_data = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    standardize_features(train_data, dataset)
    transform_targets(dataset, args.log1p_target)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    device = choose_device(args.device)
    sample = dataset[0]
    model = SNRGNN(
        node_features=sample.x.shape[1],
        edge_features=sample.edge_attr.shape[1],
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_path = args.output_dir / "best_gnn_snr.pt"

    print(
        f"Loaded {len(dataset)} graphs "
        f"({len(train_data)} train, {len(val_data)} val, {len(test_data)} test)"
    )
    print(f"Training on {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "node_features": int(sample.x.shape[1]),
                    "edge_features": int(sample.edge_attr.shape[1]),
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_mse={train_loss:.5f} "
                f"val_mse={val_loss:.5f}"
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, prediction_rows = evaluate(
        model,
        test_loader,
        device,
        use_log1p_target=args.log1p_target,
    )
    write_predictions(args.output_dir / "test_predictions.csv", prediction_rows)

    print(
        "test "
        f"mse={test_metrics['mse']:.5f} "
        f"rmse={test_metrics['rmse']:.5f} "
        f"mae={test_metrics['mae']:.5f} "
        f"r2={test_metrics['r2']:.5f}"
    )
    print(f"Saved model to {best_path}")
    print(f"Saved predictions to {args.output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
