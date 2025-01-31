import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from torch.utils.data import Dataset

ABCD_LABELS = {
    'M': 0,
    'F': 1
}


class ABCD(Dataset):
    '''A dataset class for the ABCD dataset (https://braingb.us/datasets/).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.

    LABELS_COL = 5

    ABCD_LABELS_IDX = [label for label in ABCD_LABELS.keys()]

    NUM_CLASSES = len(ABCD_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'mri', 'ABCD')
        else:
            self.root = base_root
        super().__init__()
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.x: Optional[np.ndarray] = None
        self.label: Optional[np.ndarray] = None
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        annotation_path = os.path.join(self.root, 'abcd_rest-pearson-HCP2016.npy')
        # if no data is present, prompt the user to download it
        if not os.path.exists(annotation_path):
            raise RuntimeError(
                """
                'Visit https://braingb.us/datasets/ to download the data'
                """
            )
        return annotation_path

    def build_index(self):
        print('Building index...')
        label_path = os.path.join(self.root, 'id2sex.txt')
        id_path = os.path.join(self.root, 'ids_HCP2016.txt')
        data = np.load(self.index_location, allow_pickle=True)
        with open(id_path, 'r') as f:
            ids = f.readlines()
        label_df = pd.read_csv(label_path)
        id2sex = dict(zip(label_df['id'], label_df['sex']))  # len(id2gender): 9557
        self.label = []
        self.x = []
        for id_, x in zip(ids, data):
            label = id2sex.get(id_.strip(), None)
            if label is not None:
                self.x.append(x)
                self.label.append(ABCD_LABELS[label])
        self.label = np.array(self.label)  # shape: (9557,)
        self.x = np.stack(self.x)

    def __len__(self) -> int:
        return self.label.shape[0]

    def __getitem__(self, index: int) -> Any:
        data = {
            'x': self.x[index],
            'label': self.label[index]
        }

        return data

    @staticmethod
    def num_classes():
        return ABCD.NUM_CLASSES
