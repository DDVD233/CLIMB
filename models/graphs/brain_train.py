import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Dict, Any
from heavyball import PrecondScheduleForeachSOAP
from brain_datasets import BrainNetworkDataset


def handle_nan_features(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN values in features with zeros and add a binary mask channel."""
    nan_mask = torch.isnan(x).float()
    x = torch.nan_to_num(x, nan=0.0)
    x = torch.cat([x, nan_mask], dim=-1)
    return x


class GCNConv_EdgeAttr(GCNConv):
    """Modified GCNConv that can handle edge attributes"""
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            # Modify edge weights using edge attributes
            edge_weight = edge_attr.view(-1)  # Assuming 1D edge attributes
            return super().forward(x, edge_index, edge_weight=edge_weight)
        return super().forward(x, edge_index)


class GCN_Brain(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels + in_channels  # Double for mask features

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv_EdgeAttr(self.in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv_EdgeAttr(hidden_channels, hidden_channels))

        self.convs.append(GCNConv_EdgeAttr(hidden_channels, hidden_channels))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )

        self.bn = torch.nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = handle_nan_features(x)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
        # edge_attr = torch.clamp(edge_attr, min=0)
        edge_attr = torch.abs(edge_attr)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return x


class GAT_Brain(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels + in_channels  # Double for mask features

        self.convs = torch.nn.ModuleList()
        # GAT can use edge attributes directly
        self.convs.append(GATConv(self.in_channels, hidden_channels, heads=heads, edge_dim=1))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1))

        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )

        self.bn = torch.nn.BatchNorm1d(hidden_channels * heads)

    def forward(self, x, edge_index, edge_attr, batch):
        x = handle_nan_features(x)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return x


class GraphTransformer_Brain(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels + in_channels  # Double for mask features

        self.convs = torch.nn.ModuleList()
        # TransformerConv can use edge attributes directly
        self.convs.append(TransformerConv(self.in_channels, hidden_channels, heads=heads, edge_dim=1))

        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1))

        self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )

        self.bn = torch.nn.BatchNorm1d(hidden_channels * heads)

    def forward(self, x, edge_index, edge_attr, batch):
        x = handle_nan_features(x)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return x


def safe_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate ROC AUC score with error handling."""
    try:
        if len(np.unique(y_true)) != 2:
            return 0.5

        mask = ~(np.isnan(y_true) | np.isnan(y_score).any(axis=1))
        y_true = y_true[mask]
        y_score = y_score[mask]

        if len(y_true) == 0:
            return 0.5

        return roc_auc_score(y_true, y_score[:, 1])
    except ValueError:
        return 0.5


def train_epoch(model: torch.nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                device: torch.device) -> float:
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        try:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)

            if torch.isnan(loss):
                print("Warning: NaN loss encountered, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        except RuntimeError as e:
            print(f"Warning: Error in training batch: {e}")
            continue

    return total_loss / len(loader.dataset)


def evaluate(model: torch.nn.Module,
            loader: DataLoader,
            device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            try:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = F.softmax(out, dim=1)

                if torch.isnan(pred).any():
                    print("Warning: NaN predictions encountered, skipping batch")
                    continue

                y_true.append(data.y.cpu())
                y_pred.append(pred.cpu())

            except RuntimeError as e:
                print(f"Warning: Error in evaluation batch: {e}")
                continue

    if not y_true:
        return {'auc': 0.5, 'specificity': 0.0, 'sensitivity': 0.0}

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    pred_class = y_pred.argmax(axis=1)
    mask = ~(np.isnan(y_true) | np.isnan(pred_class))
    y_true = y_true[mask]
    pred_class = pred_class[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'auc': 0.5, 'specificity': 0.0, 'sensitivity': 0.0}

    tn = np.sum((y_true == 0) & (pred_class == 0))
    tp = np.sum((y_true == 1) & (pred_class == 1))
    fn = np.sum((y_true == 1) & (pred_class == 0))
    fp = np.sum((y_true == 0) & (pred_class == 1))

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

    return {
        'auc': safe_roc_auc_score(y_true, y_pred),
        'specificity': specificity,
        'sensitivity': sensitivity
    }


def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                test_loader: DataLoader,
                epochs: int = 100,
                lr: float = 0.01,
                device: torch.device = 'cuda') -> Dict[str, Any]:

    optimizer = PrecondScheduleForeachSOAP(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_auc = 0
    best_model = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_metrics = evaluate(model, val_loader, device)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}, Val AUC: {val_metrics["auc"]:.4f}')

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    model.load_state_dict(best_model)
    test_metrics = evaluate(model, test_loader, device)

    print("\nTest set metrics:")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"Sensitivity: {test_metrics['sensitivity']:.4f}")

    return test_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_brain_networks(dataset_names: list = ['ABCD', 'PPMI', 'ABIDE']) -> Dict[str, Dict[str, Dict[str, float]]]:
    device = get_device()
    results = {}

    for dataset_name in dataset_names:
        print(f"\nTraining on {dataset_name} dataset...")

        # Load dataset
        root = 'datasets'
        train_dataset = BrainNetworkDataset(root=root, name=dataset_name, split='train')
        val_dataset = BrainNetworkDataset(root=root, name=dataset_name, split='val')
        test_dataset = BrainNetworkDataset(root=root, name=dataset_name, split='test')

        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Get dimensions from the first sample
        sample = train_dataset[0]
        in_channels = sample.x.size(1)
        hidden_channels = 64
        out_channels = 2  # Binary classification for all brain network datasets

        # Define models
        models = {
            'GCN': GCN_Brain(in_channels, hidden_channels, out_channels, num_layers=2),
            'GAT': GAT_Brain(in_channels, hidden_channels, out_channels, num_layers=2),
            'Transformer': GraphTransformer_Brain(in_channels, hidden_channels, out_channels, num_layers=2)
        }

        # Train each model
        dataset_results = {}
        for name, model in models.items():
            print(f"\nTraining {name} on {dataset_name} dataset...")
            model = model.to(device)
            dataset_results[name] = train_model(
                model, train_loader, val_loader, test_loader,
                epochs=100, device=device
            )

        results[dataset_name] = dataset_results

    # Print final results
    print("\nFinal Results:")
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name} Dataset:")
        for model_name, metrics in dataset_results.items():
            print(f"\n{model_name}:")
            print(f"AUC: {metrics['auc']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
            print(f"Sensitivity: {metrics['sensitivity']:.4f}")

    return results


if __name__ == "__main__":
    results = train_brain_networks()