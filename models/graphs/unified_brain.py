import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool
from typing import Dict, List, Tuple
from brain_datasets import BrainNetworkDataset
from torch_geometric.loader import DataLoader
from heavyball import PrecondScheduleForeachSOAP
from sklearn.metrics import roc_auc_score
import traceback


def handle_nan_features(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN values in features with zeros and add a binary mask channel."""
    nan_mask = torch.isnan(x).float()
    x = torch.nan_to_num(x, nan=0.0)
    x = torch.cat([x, nan_mask], dim=-1)
    return x


class UnifiedGAT_Brain(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 dataset_names: List[str],
                 heads: int = 4,
                 num_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels + in_channels  # Double for mask features
        self.dataset_names = dataset_names

        # Shared GAT layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(self.in_channels, hidden_channels,
                                  heads=heads, edge_dim=1))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, edge_dim=1))

        self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                  heads=1, edge_dim=1))

        # Separate classifiers for each dataset
        self.classifiers = torch.nn.ModuleDict({
            name: torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.GELU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(hidden_channels, 2)  # Binary classification per dataset
            ) for name in dataset_names
        })

        self.bn = torch.nn.BatchNorm1d(hidden_channels * heads)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                dataset_name: str) -> torch.Tensor:
        x = handle_nan_features(x)
        # pad x to in_channels
        if x.size(-1) < self.in_channels:
            x = F.pad(x, (0, self.in_channels - x.size(-1)), value=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)

        # Shared feature extraction
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Dataset-specific classification
        return self.classifiers[dataset_name](x)


class UnifiedGraphTransformer_Brain(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 dataset_names: List[str],
                 heads: int = 4,
                 num_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels + in_channels  # Double for mask features
        self.dataset_names = dataset_names

        # Shared Transformer layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(self.in_channels, hidden_channels,
                                          heads=heads, edge_dim=1))

        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels,
                                              heads=heads, edge_dim=1))

        self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels,
                                          heads=1, edge_dim=1))

        # Separate classifiers for each dataset
        self.classifiers = torch.nn.ModuleDict({
            name: torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(hidden_channels, 2)  # Binary classification per dataset
            ) for name in dataset_names
        })

        self.bn = torch.nn.BatchNorm1d(hidden_channels * heads)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                dataset_name: str) -> torch.Tensor:
        x = handle_nan_features(x)
        # pad x to in_channels
        if x.size(-1) < self.in_channels:
            x = F.pad(x, (0, self.in_channels - x.size(-1)), value=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)

        # Shared feature extraction
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Dataset-specific classification
        return self.classifiers[dataset_name](x)


# Modified training function to handle dataset mixture
def train_epoch_unified(model: torch.nn.Module,
                        loaders: Dict[str, DataLoader],
                        optimizer: torch.optim.Optimizer,
                        criterion: torch.nn.Module,
                        device: torch.device) -> float:
    model.train()
    total_loss = 0
    total_samples = sum(len(loader.dataset) for loader in loaders.values())

    # Create iterators for all loaders
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    active_datasets = set(loaders.keys())

    while active_datasets:
        for dataset_name in list(active_datasets):
            try:
                data = next(iterators[dataset_name]).to(device)
                optimizer.zero_grad()

                out = model(data.x, data.edge_index, data.edge_attr,
                            data.batch, dataset_name)
                loss = criterion(out, data.y)

                if torch.isnan(loss):
                    print(f"Warning: NaN loss encountered in {dataset_name}, skipping batch")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            except StopIteration:
                active_datasets.remove(dataset_name)
            except RuntimeError as e:
                print(f"Warning: Error in training batch for {dataset_name}: {e}")
                print(traceback.format_exc())
                continue

    return total_loss / total_samples


def evaluate_unified(model: torch.nn.Module,
                     loaders: Dict[str, DataLoader],
                     device: torch.device) -> Dict[str, Dict[str, float]]:
    model.eval()
    results = {}

    for dataset_name, loader in loaders.items():
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                try:
                    out = model(data.x, data.edge_index, data.edge_attr,
                                data.batch, dataset_name)
                    pred = F.softmax(out, dim=1)

                    if torch.isnan(pred).any():
                        print(f"Warning: NaN predictions encountered in {dataset_name}, skipping batch")
                        continue

                    y_true.append(data.y.cpu())
                    y_pred.append(pred.cpu())

                except RuntimeError as e:
                    print(f"Warning: Error in evaluation batch for {dataset_name}: {e}")
                    continue

        if not y_true:
            results[dataset_name] = {'auc': 0.5, 'specificity': 0.0, 'sensitivity': 0.0}
            continue

        # Calculate metrics for each dataset separately
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        # Calculate metrics
        pred_class = y_pred.argmax(axis=1)
        mask = ~(np.isnan(y_true) | np.isnan(pred_class))
        y_true = y_true[mask]
        pred_class = pred_class[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            results[dataset_name] = {'auc': 0.5, 'specificity': 0.0, 'sensitivity': 0.0}
            continue

        # Calculate confusion matrix metrics
        tn = np.sum((y_true == 0) & (pred_class == 0))
        tp = np.sum((y_true == 1) & (pred_class == 1))
        fn = np.sum((y_true == 1) & (pred_class == 0))
        fp = np.sum((y_true == 0) & (pred_class == 1))

        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

        # Calculate AUC
        try:
            if len(np.unique(y_true)) != 2:
                auc = 0.5
            else:
                auc = roc_auc_score(y_true, y_pred[:, 1])
        except ValueError:
            auc = 0.5

        results[dataset_name] = {
            'auc': auc,
            'specificity': specificity,
            'sensitivity': sensitivity
        }

        return results


def train_unified_model(model: torch.nn.Module,
                        train_loaders: Dict[str, DataLoader],
                        val_loaders: Dict[str, DataLoader],
                        test_loaders: Dict[str, DataLoader],
                        epochs: int = 100,
                        lr: float = 0.01,
                        device: torch.device = 'cuda') -> Dict[str, Dict[str, float]]:
    optimizer = PrecondScheduleForeachSOAP(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_score = 0
    best_model = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        loss = train_epoch_unified(model, train_loaders, optimizer, criterion, device)

        # Evaluate on validation set for each dataset
        val_metrics = evaluate_unified(model, val_loaders, device)

        # Calculate average validation AUC across all datasets
        avg_val_auc = sum(metrics['auc'] for metrics in val_metrics.values()) / len(val_metrics)

        if avg_val_auc > best_val_score:
            best_val_score = avg_val_auc
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}')
            for dataset_name, metrics in val_metrics.items():
                print(f'{dataset_name} Val AUC: {metrics["auc"]:.4f}')

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    test_metrics = evaluate_unified(model, test_loaders, device)

    return test_metrics


def train_unified_brain_networks(dataset_names: list = ['ABIDE', 'ABCD', 'PPMI']) -> Dict[
    str, Dict[str, Dict[str, float]]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    # Load all datasets
    loaders = {}
    for split in ['train', 'val', 'test']:
        loaders[split] = {}
        for dataset_name in dataset_names:
            dataset = BrainNetworkDataset(root='datasets', name=dataset_name, split=split)
            batch_size = 32
            shuffle = (split == 'train')
            loaders[split][dataset_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Get dimensions from the first sample
    # sample = next(iter(loaders['train'][dataset_names[0]]))
    in_channels = 360
    hidden_channels = 64

    # Define unified models
    models = {
        'UnifiedGAT': UnifiedGAT_Brain(in_channels, hidden_channels, dataset_names, num_layers=2),
        'UnifiedTransformer': UnifiedGraphTransformer_Brain(in_channels, hidden_channels, dataset_names, num_layers=2)
    }

    # Train each unified model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model = model.to(device)

        results[name] = train_unified_model(
            model,
            loaders['train'],
            loaders['val'],
            loaders['test'],
            epochs=100,
            device=device
        )

    # Print final results
    print(results)

    return results


if __name__ == "__main__":
    results = train_unified_brain_networks()