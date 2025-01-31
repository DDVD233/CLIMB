import torch
from ogb.graphproppred import GraphPropPredDataset
import pickle
import numpy as np
import os
from pathlib import Path


def create_directory(path):
    """
    Create directory if it doesn't exist
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def convert_graph_to_dict(graph_dict, label):
    """
    Convert OGB graph dictionary to our desired dictionary format
    """
    processed_dict = {
        'edge_index': graph_dict['edge_index'],
        'node_features': graph_dict['node_feat'] if 'node_feat' in graph_dict else None,
        'edge_features': graph_dict['edge_feat'] if 'edge_feat' in graph_dict else None,
        'label': label
    }

    return processed_dict


def process_ogbg_molhiv(save_dir='/mnt/8T/high_modality/molecule/ogbg_molhiv'):
    # Create the save directory if it doesn't exist
    create_directory(save_dir)

    # Download and load the dataset
    print(f"Downloading and loading ogbg-molhiv dataset to {save_dir}...")
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=os.path.join(save_dir, 'raw'))
    split_idx = dataset.get_idx_split()

    # Convert each graph to dictionary format
    print("Converting graphs to dictionary format...")
    graph_dicts = []

    for idx in range(len(dataset)):
        if idx % 1000 == 0:
            print(f"Processing graph {idx}/{len(dataset)}")

        graph_dict, label = dataset[idx]
        processed_dict = convert_graph_to_dict(graph_dict, label)

        # Add metadata
        processed_dict['graph_id'] = idx

        # Determine which split this graph belongs to
        if idx in split_idx['train']:
            processed_dict['split'] = 'train'
        elif idx in split_idx['valid']:
            processed_dict['split'] = 'valid'
        else:
            processed_dict['split'] = 'test'

        graph_dicts.append(processed_dict)

    # Save to pickle file
    processed_dir = os.path.join(save_dir, 'processed')
    create_directory(processed_dir)
    output_file = os.path.join(processed_dir, 'ogbg_molhiv_processed.pkl')
    print(f"Saving processed dataset to {output_file}...")

    with open(output_file, 'wb') as f:
        pickle.dump(graph_dicts, f)

    print("Processing completed!")

    # Print some statistics and save them to a log file
    stats = "\nDataset statistics:\n"
    stats += f"Total number of graphs: {len(graph_dicts)}\n"
    stats += f"Number of node features: {graph_dicts[0]['node_features'].shape[1]}\n"
    if graph_dicts[0]['edge_features'] is not None:
        stats += f"Number of edge features: {graph_dicts[0]['edge_features'].shape[1]}\n"

    # Add label statistics
    labels = np.array([g['label'] for g in graph_dicts])
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)
    stats += f"\nLabel distribution:\n"
    stats += f"Positive samples: {pos_count} ({pos_count/len(labels)*100:.2f}%)\n"
    stats += f"Negative samples: {neg_count} ({neg_count/len(labels)*100:.2f}%)\n"

    # Add split statistics
    split_counts = {
        'train': sum(1 for g in graph_dicts if g['split'] == 'train'),
        'valid': sum(1 for g in graph_dicts if g['split'] == 'valid'),
        'test': sum(1 for g in graph_dicts if g['split'] == 'test')
    }
    stats += f"\nSplit sizes:\n"
    stats += f"Train: {split_counts['train']}\n"
    stats += f"Valid: {split_counts['valid']}\n"
    stats += f"Test: {split_counts['test']}\n"

    print(stats)

    # Save statistics to a log file
    log_file = os.path.join(processed_dir, 'dataset_stats.txt')
    with open(log_file, 'w') as f:
        f.write(stats)


if __name__ == "__main__":
    process_ogbg_molhiv()