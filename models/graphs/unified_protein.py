import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATv2Conv, global_mean_pool
from torch_geometric.datasets import PPI, TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
import numpy as np


class UnifiedGraphTransformer(torch.nn.Module):
    def __init__(self, in_channels_ppi, in_channels_proteins,
                 hidden_channels, out_channels_ppi, out_channels_proteins,
                 heads=4, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        # Separate input projections for different feature dimensions
        self.ppi_proj = torch.nn.Linear(in_channels_ppi, hidden_channels)
        self.proteins_proj = torch.nn.Linear(in_channels_proteins, hidden_channels)

        # Shared transformer layers
        self.transformers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.transformers.append(TransformerConv(
                hidden_channels * heads if _ > 0 else hidden_channels,
                hidden_channels,
                heads=heads,
                dropout=0.1
            ))

        # Task-specific classifiers
        # For PPI (node-level)
        self.ppi_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels_ppi)
        )

        # For PROTEINS (graph-level)
        self.proteins_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels_proteins)
        )

    def forward(self, x, edge_index, batch=None, task='ppi'):
        # Project input features based on task
        if task == 'ppi':
            x = self.ppi_proj(x)
        else:  # proteins
            x = self.proteins_proj(x)

        # Shared transformer layers
        for transformer in self.transformers[:-1]:
            x = transformer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # Last transformer layer
        x = self.transformers[-1](x, edge_index)

        # Task-specific predictions
        if task == 'ppi':
            # Node-level prediction for PPI
            return self.ppi_classifier(x)
        else:
            # Graph-level prediction for PROTEINS
            # First pool the node embeddings
            x = global_mean_pool(x, batch)
            return self.proteins_classifier(x)


class UnifiedGAT(torch.nn.Module):
    def __init__(self, in_channels_ppi, in_channels_proteins,
                 hidden_channels, out_channels_ppi, out_channels_proteins,
                 heads=4, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        # Separate input projections
        self.ppi_proj = torch.nn.Linear(in_channels_ppi, hidden_channels)
        self.proteins_proj = torch.nn.Linear(in_channels_proteins, hidden_channels)

        # Shared GAT layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                hidden_channels * heads if _ > 0 else hidden_channels,
                hidden_channels,
                heads=heads,
                dropout=0.1
            ))

        # Task-specific classifiers
        # For PPI (node-level)
        self.ppi_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels_ppi)
        )

        # For PROTEINS (graph-level)
        self.proteins_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels_proteins)
        )

    def forward(self, x, edge_index, batch=None, task='ppi'):
        # Project input features based on task
        if task == 'ppi':
            x = self.ppi_proj(x)
        else:  # proteins
            x = self.proteins_proj(x)

        # Shared GAT layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # Last GAT layer
        x = self.convs[-1](x, edge_index)

        # Task-specific predictions
        if task == 'ppi':
            return self.ppi_classifier(x)
        else:
            x = global_mean_pool(x, batch)
            return self.proteins_classifier(x)


def train_epoch_unified(model, ppi_loader, proteins_loader, optimizer, device):
    model.train()
    total_loss = 0
    num_graphs = 0

    # Create iterator for proteins loader
    proteins_iter = iter(proteins_loader)

    # Train on PPI data
    for ppi_data in ppi_loader:
        # Get proteins batch if available
        try:
            proteins_data = next(proteins_iter)
        except StopIteration:
            proteins_iter = iter(proteins_loader)
            proteins_data = next(proteins_iter)

        # Move data to device
        ppi_data = ppi_data.to(device)
        proteins_data = proteins_data.to(device)

        optimizer.zero_grad()

        # Forward pass for PPI
        ppi_out = model(ppi_data.x, ppi_data.edge_index, task='ppi')
        ppi_loss = F.binary_cross_entropy_with_logits(ppi_out, ppi_data.y)

        # Forward pass for PROTEINS
        proteins_out = model(proteins_data.x, proteins_data.edge_index,
                             proteins_data.batch, task='proteins')
        proteins_loss = F.cross_entropy(proteins_out, proteins_data.y.squeeze())

        # Combined loss
        loss = ppi_loss + proteins_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_graphs += ppi_data.num_graphs + proteins_data.num_graphs

    return total_loss / num_graphs


def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def evaluate_unified(model, loader, device, task='ppi'):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            if task == 'ppi':
                out = model(data.x, data.edge_index, task='ppi')
                pred = torch.sigmoid(out)
            else:
                out = model(data.x, data.edge_index, data.batch, task='proteins')
                pred = F.softmax(out, dim=1)

            ys.append(data.y.cpu())
            preds.append(pred.cpu())

    y = torch.cat(ys, dim=0).numpy()
    pred = torch.cat(preds, dim=0).numpy()

    if task == 'ppi':
        # Multi-label metrics for PPI
        auc_scores = []
        sensitivities = []
        specificities = []

        for i in range(y.shape[1]):
            if len(np.unique(y[:, i])) > 1:
                # Calculate binary predictions using 0.5 threshold
                binary_preds = (pred[:, i] > 0.5).astype(int)

                # Calculate metrics
                auc_scores.append(roc_auc_score(y[:, i], pred[:, i]))
                sensitivities.append(recall_score(y[:, i], binary_preds))
                specificities.append(calculate_specificity(y[:, i], binary_preds))

        return {
            'auc': np.mean(auc_scores),
            'sensitivity': np.mean(sensitivities),
            'specificity': np.mean(specificities)
        }
    else:
        # Binary classification metrics for PROTEINS
        binary_preds = np.argmax(pred, axis=1)
        return {
            'auc': roc_auc_score(y, pred[:, 1]),
            'sensitivity': recall_score(y, binary_preds),
            'specificity': calculate_specificity(y, binary_preds)
        }


def train_unified_model(model, model_name, ppi_train_loader, ppi_val_loader, ppi_test_loader,
                        proteins_train_loader, proteins_val_loader, proteins_test_loader,
                        epochs=100, lr=0.001, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_score = 0
    best_model = None

    for epoch in range(epochs):
        # Training
        loss = train_epoch_unified(model, ppi_train_loader, proteins_train_loader,
                                   optimizer, device)

        # Validation
        ppi_val_metrics = evaluate_unified(model, ppi_val_loader, device, task='ppi')
        proteins_val_metrics = evaluate_unified(model, proteins_val_loader, device,
                                                task='proteins')
        val_score = (ppi_val_metrics['auc'] + proteins_val_metrics['auc']) / 2

        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f'{model_name} - Epoch {epoch + 1:03d}, Loss: {loss:.4f}')
            print(f'PPI Val - AUC: {ppi_val_metrics["auc"]:.4f}, '
                  f'Sensitivity: {ppi_val_metrics["sensitivity"]:.4f}, '
                  f'Specificity: {ppi_val_metrics["specificity"]:.4f}')
            print(f'PROTEINS Val - AUC: {proteins_val_metrics["auc"]:.4f}, '
                  f'Sensitivity: {proteins_val_metrics["sensitivity"]:.4f}, '
                  f'Specificity: {proteins_val_metrics["specificity"]:.4f}')

    # Load best model and evaluate
    model.load_state_dict(best_model)
    ppi_test_metrics = evaluate_unified(model, ppi_test_loader, device, task='ppi')
    proteins_test_metrics = evaluate_unified(model, proteins_test_loader, device,
                                             task='proteins')

    return {
        'ppi': ppi_test_metrics,
        'proteins': proteins_test_metrics
    }


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load PPI dataset
    print("\nLoading PPI dataset...")
    train_dataset_ppi = PPI(root='/tmp/PPI', split='train')
    val_dataset_ppi = PPI(root='/tmp/PPI', split='val')
    test_dataset_ppi = PPI(root='/tmp/PPI', split='test')

    # Load PROTEINS dataset
    print("\nLoading PROTEINS dataset...")
    dataset_proteins = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')

    # Split PROTEINS dataset
    train_size = int(0.8 * len(dataset_proteins))
    val_size = int(0.1 * len(dataset_proteins))
    test_size = len(dataset_proteins) - train_size - val_size

    train_dataset_proteins, val_dataset_proteins, test_dataset_proteins = torch.utils.data.random_split(
        dataset_proteins, [train_size, val_size, test_size])

    # Create dataloaders
    batch_size = 4
    train_loader_ppi = DataLoader(train_dataset_ppi, batch_size=batch_size, shuffle=True)
    val_loader_ppi = DataLoader(val_dataset_ppi, batch_size=batch_size, shuffle=False)
    test_loader_ppi = DataLoader(test_dataset_ppi, batch_size=batch_size, shuffle=False)

    train_loader_proteins = DataLoader(train_dataset_proteins, batch_size=batch_size, shuffle=True)
    val_loader_proteins = DataLoader(val_dataset_proteins, batch_size=batch_size, shuffle=False)
    test_loader_proteins = DataLoader(test_dataset_proteins, batch_size=batch_size, shuffle=False)

    # Model configs
    models = {
        # 'GAT': UnifiedGAT(
        #     in_channels_ppi=train_dataset_ppi.num_features,
        #     in_channels_proteins=dataset_proteins.num_features,
        #     hidden_channels=256,
        #     out_channels_ppi=train_dataset_ppi.num_classes,
        #     out_channels_proteins=dataset_proteins.num_classes,
        #     heads=4,
        #     num_layers=3
        # ),
        'Transformer': UnifiedGraphTransformer(
            in_channels_ppi=train_dataset_ppi.num_features,
            in_channels_proteins=dataset_proteins.num_features,
            hidden_channels=256,
            out_channels_ppi=train_dataset_ppi.num_classes,
            out_channels_proteins=dataset_proteins.num_classes,
            heads=4,
            num_layers=2
        )
    }

    print("\nStarting training...")
    print(
        f"PPI dataset size - Train: {len(train_dataset_ppi)}, Val: {len(val_dataset_ppi)}, Test: {len(test_dataset_ppi)}")
    print(
        f"PROTEINS dataset size - Train: {len(train_dataset_proteins)}, Val: {len(val_dataset_proteins)}, Test: {len(test_dataset_proteins)}")

    # Train all models
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model = model.to(device)
        results[model_name] = train_unified_model(
            model=model,
            model_name=model_name,
            ppi_train_loader=train_loader_ppi,
            ppi_val_loader=val_loader_ppi,
            ppi_test_loader=test_loader_ppi,
            proteins_train_loader=train_loader_proteins,
            proteins_val_loader=val_loader_proteins,
            proteins_test_loader=test_loader_proteins,
            epochs=100,
            lr=0.001,
            device=device
        )

    # Print final results
    print("\nFinal Results:")
    print(results)