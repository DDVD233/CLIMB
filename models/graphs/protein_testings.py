import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_add_pool
from torch_geometric.datasets import PPI, TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from heavyball import PrecondScheduleForeachSOAP


# Node-level classification for PPI
class GCN_Node(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# Graph-level classification for PROTEINS
class GCN_Graph(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels))  # Last conv layer keeps hidden dim

        # MLP for graph-level prediction after pooling
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        # Node embedding
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)

        # Graph pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        # Graph-level prediction
        x = self.mlp(x)
        return x


# Node-level classification for PPI
class GAT_Node(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# Graph-level classification for PROTEINS
class GAT_Graph(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))

        # MLP for graph-level prediction after pooling
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        # Node embedding
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)

        # Graph pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        # Graph-level prediction
        x = self.mlp(x)
        return x


# Node-level classification for PPI
class GraphTransformer_Node(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads))

        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.convs.append(TransformerConv(hidden_channels * heads, out_channels, heads=1))

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# Graph-level classification for PROTEINS
class GraphTransformer_Graph(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads))

        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=1))

        # MLP for graph-level prediction after pooling
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        # Node embedding
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)

        # Graph pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        # Graph-level prediction
        x = self.mlp(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device, is_multilabel=False):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if is_multilabel:
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index, data.batch)

        if is_multilabel:
            loss = criterion(out, data.y)
        else:
            loss = criterion(out, data.y.squeeze())

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, is_multilabel=False):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            if is_multilabel:
                out = model(data.x, data.edge_index)
            else:
                out = model(data.x, data.edge_index, data.batch)

            if is_multilabel:
                pred = torch.sigmoid(out)
            else:
                pred = F.softmax(out, dim=1)

            ys.append(data.y.cpu())
            preds.append(pred.cpu())

    y = torch.cat(ys, dim=0).numpy()
    pred = torch.cat(preds, dim=0).numpy()

    if is_multilabel:
        # For multi-label classification (PPI)
        auc_scores = []
        specificities = []
        sensitivities = []

        for i in range(y.shape[1]):
            if len(np.unique(y[:, i])) > 1:  # Only evaluate if both classes present
                auc_scores.append(roc_auc_score(y[:, i], pred[:, i]))
                prec, rec, _ = precision_recall_curve(y[:, i], pred[:, i])
                specificities.append(np.mean(prec))
                sensitivities.append(np.mean(rec))

        return {
            'auc': np.mean(auc_scores),
            'specificity': np.mean(specificities),
            'sensitivity': np.mean(sensitivities)
        }
    else:
        # For binary classification (PROTEINS)
        # y_bin = y.argmax(axis=1)
        pred_bin = pred.argmax(axis=1)

        tn = np.sum((y == 0) & (pred_bin == 0))
        tp = np.sum((y == 1) & (pred_bin == 1))
        fn = np.sum((y == 1) & (pred_bin == 0))
        fp = np.sum((y == 0) & (pred_bin == 1))

        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

        return {
            'auc': roc_auc_score(y, pred[:, 1]),
            'specificity': specificity,
            'sensitivity': sensitivity
        }


def train_model(model, train_loader, val_loader, test_loader, epochs=100, lr=0.01,
                device='cuda', is_multilabel=False):
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = PrecondScheduleForeachSOAP(model.parameters(), lr=lr)

    if is_multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_val_auc = 0
    best_model = None

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, is_multilabel)

        # Validation
        val_metrics = evaluate(model, val_loader, device, is_multilabel)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}, Val AUC: {val_metrics["auc"]:.4f}')

    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    test_metrics = evaluate(model, test_loader, device, is_multilabel)

    print("\nTest set metrics:")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"Sensitivity: {test_metrics['sensitivity']:.4f}")

    return test_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# Example usage for PPI dataset
def train_ppi():
    device = get_device()

    # Load PPI dataset
    train_dataset = PPI(root='/tmp/PPI', split='train')
    val_dataset = PPI(root='/tmp/PPI', split='val')
    test_dataset = PPI(root='/tmp/PPI', split='test')

    bs = 64
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # Define hyperparameters
    in_channels = train_dataset.num_features
    hidden_channels = 256
    out_channels = train_dataset.num_classes

    # Train different models
    models = {
        'GCN': GCN_Node(in_channels, hidden_channels, out_channels),
        'GAT': GAT_Node(in_channels, hidden_channels, out_channels),
        'Transformer': GraphTransformer_Node(in_channels, hidden_channels, out_channels)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} on PPI dataset...")
        model = model.to(device)
        results[name] = train_model(model, train_loader, val_loader, test_loader,
                                    epochs=300, device=device, is_multilabel=True)

    return results


# Example usage for PROTEINS dataset
def train_proteins():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load PROTEINS dataset
    dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define hyperparameters
    in_channels = dataset.num_features
    hidden_channels = 64
    out_channels = dataset.num_classes

    # Train different models (now using graph-level models)
    models = {
        'GCN': GCN_Graph(in_channels, hidden_channels, out_channels),
        'GAT': GAT_Graph(in_channels, hidden_channels, out_channels),
        'Transformer': GraphTransformer_Graph(in_channels, hidden_channels, out_channels)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} on PROTEINS dataset...")
        model = model.to(device)
        results[name] = train_model(model, train_loader, val_loader, test_loader,
                                    epochs=100, device=device, is_multilabel=False)

    return results


if __name__ == "__main__":
    print("Training on PPI dataset...")
    ppi_results = train_ppi()

    print("\nTraining on PROTEINS dataset...")
    proteins_results = train_proteins()

    # Print final results
    print("\nFinal Results:")
    print("\nPPI Dataset:")
    for model, metrics in ppi_results.items():
        print(f"\n{model}:")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")

    print("\nPROTEINS Dataset:")
    for model, metrics in proteins_results.items():
        print(f"\n{model}:")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")