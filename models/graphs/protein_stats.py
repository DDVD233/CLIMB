import torch
from torch_geometric.datasets import PPI, TUDataset
from torch_geometric.loader import DataLoader


def analyze_ppi_dataset():
    """Analyze the PPI dataset statistics"""
    # Load all splits of PPI dataset
    train_dataset = PPI(root='/tmp/PPI', split='train')
    val_dataset = PPI(root='/tmp/PPI', split='val')
    test_dataset = PPI(root='/tmp/PPI', split='test')

    # Calculate statistics
    stats = {
        'train_graphs': len(train_dataset),
        'val_graphs': len(val_dataset),
        'test_graphs': len(test_dataset),
        'total_graphs': len(train_dataset) + len(val_dataset) + len(test_dataset),
        'features': train_dataset.num_features,
        'classes': train_dataset.num_classes,
    }

    # Calculate total nodes and edges
    train_nodes = sum(data.num_nodes for data in train_dataset)
    train_edges = sum(data.edge_index.shape[1] for data in train_dataset)
    val_nodes = sum(data.num_nodes for data in val_dataset)
    val_edges = sum(data.edge_index.shape[1] for data in val_dataset)
    test_nodes = sum(data.num_nodes for data in test_dataset)
    test_edges = sum(data.edge_index.shape[1] for data in test_dataset)

    stats.update({
        'train_nodes': train_nodes,
        'train_edges': train_edges,
        'val_nodes': val_nodes,
        'val_edges': val_edges,
        'test_nodes': test_nodes,
        'test_edges': test_edges,
        'total_nodes': train_nodes + val_nodes + test_nodes,
        'total_edges': train_edges + val_edges + test_edges
    })

    return stats


def analyze_proteins_dataset():
    """Analyze the PROTEINS dataset statistics"""
    # Load PROTEINS dataset
    dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')

    # Calculate total nodes and edges
    total_nodes = sum(data.num_nodes for data in dataset)
    total_edges = sum(data.edge_index.shape[1] for data in dataset)

    # Calculate statistics
    stats = {
        'total_graphs': len(dataset),
        'features': dataset.num_features,
        'classes': dataset.num_classes,
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'avg_nodes_per_graph': total_nodes / len(dataset),
        'avg_edges_per_graph': total_edges / len(dataset)
    }

    return stats


def print_dataset_stats():
    """Print comprehensive statistics for both datasets"""
    print("Analyzing datasets...")

    # Analyze PPI dataset
    print("\nPPI Dataset Statistics:")
    ppi_stats = analyze_ppi_dataset()
    print(f"Number of graphs:")
    print(f"  Train: {ppi_stats['train_graphs']}")
    print(f"  Validation: {ppi_stats['val_graphs']}")
    print(f"  Test: {ppi_stats['test_graphs']}")
    print(f"  Total: {ppi_stats['total_graphs']}")
    print(f"\nNumber of nodes:")
    print(f"  Train: {ppi_stats['train_nodes']:,}")
    print(f"  Validation: {ppi_stats['val_nodes']:,}")
    print(f"  Test: {ppi_stats['test_nodes']:,}")
    print(f"  Total: {ppi_stats['total_nodes']:,}")
    print(f"\nNumber of edges:")
    print(f"  Train: {ppi_stats['train_edges']:,}")
    print(f"  Validation: {ppi_stats['val_edges']:,}")
    print(f"  Test: {ppi_stats['test_edges']:,}")
    print(f"  Total: {ppi_stats['total_edges']:,}")
    print(f"\nFeatures per node: {ppi_stats['features']}")
    print(f"Number of classes: {ppi_stats['classes']}")

    # Analyze PROTEINS dataset
    print("\nPROTEINS Dataset Statistics:")
    proteins_stats = analyze_proteins_dataset()
    print(f"Total number of graphs: {proteins_stats['total_graphs']}")
    print(f"Total number of nodes: {proteins_stats['total_nodes']:,}")
    print(f"Total number of edges: {proteins_stats['total_edges']:,}")
    print(f"Average nodes per graph: {proteins_stats['avg_nodes_per_graph']:.2f}")
    print(f"Average edges per graph: {proteins_stats['avg_edges_per_graph']:.2f}")
    print(f"Features per node: {proteins_stats['features']}")
    print(f"Number of classes: {proteins_stats['classes']}")


if __name__ == "__main__":
    print_dataset_stats()