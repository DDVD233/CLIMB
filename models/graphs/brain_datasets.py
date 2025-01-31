import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
from scipy.io import loadmat
from typing import Optional, Tuple, List
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


class BrainNetworkDataset(Dataset):
    """
    A universal PyTorch Geometric dataset class for brain network datasets (ABCD, PPMI, ABIDE)

    Args:
        root (str): Root directory of the dataset
        name (str): Name of the dataset ('ABCD', 'PPMI', or 'ABIDE')
        split (str): Data split ('train', 'val', or 'test')
        transform (optional): Transform to be applied to the data
    """

    def __init__(
            self,
            root: str,
            name: str,
            split: str = 'train',
            transform: Optional[object] = None,
            train_ratio: float = 0.7,
            val_ratio: float = 0.1
    ):
        self.name = name.upper()
        assert self.name in ['ABCD', 'PPMI', 'ABIDE'], f"Dataset {name} not supported"
        assert split in ['train', 'val', 'test'], f"Split {split} not supported"

        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        super().__init__(root, transform)

        # Load and process the data
        self.data_list = self._process()

    def _process_abcd(self) -> List[Data]:
        """Process ABCD dataset"""
        # Load timeseries and correlation data
        pearson = np.load(os.path.join(self.root, 'ABCD', 'abcd_rest-pearson-HCP2016.npy'))

        # Load IDs and labels
        with open(os.path.join(self.root, 'ABCD', 'id2sex.txt'), 'r') as f:
            label_data = pd.read_csv(f)

        # Create label encoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(label_data['sex'])

        data_list = []
        for idx in range(len(pearson)):
            # Create adjacency matrix and edge information
            adj_matrix = pearson[idx]
            edge_index, edge_attr = self._adj_to_edge_index(adj_matrix)

            # Use rows of adjacency matrix as node features
            x = torch.FloatTensor(adj_matrix)

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(labels[idx], dtype=torch.long)
            )
            data_list.append(data)

        return data_list

    def _process_ppmi(self) -> List[Data]:
        """Process PPMI dataset"""
        # Load .mat file
        mat_data = loadmat(os.path.join(self.root, 'PPMI', 'PPMI.mat'))

        # Extract labels and features
        labels = mat_data['label'].reshape(-1)
        labels[labels == -1] = 0  # Convert -1 to 0

        # Get correlation matrices (first view)
        correlation_matrices = mat_data['X']
        data_list = []

        for idx in range(correlation_matrices.shape[0]):
            # Get correlation matrix for this sample (first view)
            adj_matrix = correlation_matrices[idx][0][:, :, 0]
            edge_index, edge_attr = self._adj_to_edge_index(adj_matrix)

            # Use rows of adjacency matrix as node features
            x = torch.FloatTensor(adj_matrix)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(labels[idx], dtype=torch.long)
            )
            data_list.append(data)

        return data_list

    def _process_abide(self) -> List[Data]:
        """Process ABIDE dataset"""
        # Load the .npy file
        data_dict = np.load(os.path.join(self.root, 'ABIDE', 'abide.npy'), allow_pickle=True).item()

        # Extract correlation matrices and labels
        correlation_matrices = data_dict['corr']
        labels = data_dict['label']
        sites = data_dict['site']  # Site information for stratification

        data_list = []
        site_index_mapping = {site: idx for idx, site in enumerate(np.unique(sites))}
        for idx in range(len(correlation_matrices)):
            adj_matrix = correlation_matrices[idx]
            edge_index, edge_attr = self._adj_to_edge_index(adj_matrix)

            # Use rows of adjacency matrix as node features
            x = torch.FloatTensor(adj_matrix)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(labels[idx], dtype=torch.long),
                site=torch.tensor(site_index_mapping[sites[idx]], dtype=torch.long)
            )
            data_list.append(data)

        return data_list

    def _adj_to_edge_index(self, adj_matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert adjacency matrix to edge index and edge attributes"""
        # Get indices where values are non-zero
        rows, cols = np.nonzero(adj_matrix)
        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)

        # Get the corresponding values as edge attributes
        edge_attr = torch.tensor(adj_matrix[rows, cols], dtype=torch.float).reshape(-1, 1)

        return edge_index, edge_attr

    def _split_data(self, data_list: List[Data]) -> List[Data]:
        """Split data into train/val/test sets"""
        n_samples = len(data_list)

        if self.name == 'ABIDE':
            # Stratified split for ABIDE using site information
            sites = [data.site.item() for data in data_list]
            indices = np.arange(n_samples)

            # First split: train vs (val+test)
            splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=42)
            train_idx, temp_idx = next(splitter1.split(indices, sites))

            # Second split: val vs test
            val_test_ratio = self.val_ratio / (1 - self.train_ratio)
            splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_test_ratio, random_state=42)
            val_idx, test_idx = next(splitter2.split(indices[temp_idx], np.array(sites)[temp_idx]))
            val_idx = temp_idx[val_idx]
            test_idx = temp_idx[test_idx]

        else:
            # Random split for other datasets
            indices = torch.randperm(n_samples)
            train_size = int(n_samples * self.train_ratio)
            val_size = int(n_samples * self.val_ratio)

            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]

        if self.split == 'train':
            return [data_list[i] for i in train_idx]
        elif self.split == 'val':
            return [data_list[i] for i in val_idx]
        else:  # test
            return [data_list[i] for i in test_idx]

    def _process(self) -> List[Data]:
        """Process the dataset based on the name"""
        if self.name == 'ABCD':
            data_list = self._process_abcd()
        elif self.name == 'PPMI':
            data_list = self._process_ppmi()
        else:  # ABIDE
            data_list = self._process_abide()

        return self._split_data(data_list)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]