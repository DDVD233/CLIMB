import os
from typing import Any, List, Optional

import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

PPMI_LABELS = {
    'Negative': 0,
    'Positive': 1
}


class PPMI(Dataset):
    '''A dataset class for the PPMI dataset (https://braingb.us/datasets/).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.

    LABELS_COL = 5

    PPMI_LABELS_IDX = [label for label in PPMI_LABELS.keys()]

    NUM_CLASSES = len(PPMI_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'mri', 'PPMI')
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
        annotation_path = os.path.join(self.root, 'PPMI.mat')
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
        data = sio.loadmat(self.index_location)
        self.x = np.stack(list(data['X'].squeeze()))
        self.label = np.stack(list(data['label'].squeeze()))

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
        return PPMI.NUM_CLASSES
