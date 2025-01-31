import os
from typing import Any, List, Optional

import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

ABIDE_LABELS = {
    'Negative': 0,
    'Positive': 1
}


class ABIDE(Dataset):
    '''A dataset class for the PPMI dataset (https://braingb.us/datasets/).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.

    LABELS_COL = 5

    ABIDE_LABELS_IDX = [label for label in ABIDE_LABELS.keys()]

    NUM_CLASSES = len(ABIDE_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'mri', 'ABIDE')
        else:
            self.root = base_root
        super().__init__()
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.x: Optional[np.ndarray] = None
        self.label: Optional[np.ndarray] = None
        self.site = None
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        annotation_path = os.path.join(self.root, 'ABIDE.npy')
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
        data = np.load(self.index_location, allow_pickle=True).item()
        self.x = data['corr']
        self.label = data['label']
        self.site = data['site']

    def __len__(self) -> int:
        return self.label.shape[0]

    def __getitem__(self, index: int) -> Any:
        data = {
            'x': self.x[index],
            'label': int(self.label[index]),
            'site': self.site[index]
        }

        return data

    @staticmethod
    def num_classes():
        return ABIDE.NUM_CLASSES
