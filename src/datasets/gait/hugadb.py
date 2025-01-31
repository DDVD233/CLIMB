import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from torch.utils.data import Dataset

HuGaDB_LABELS = {
    'Walking': 1,
    'Running': 2,
    'Going Up': 3,
    'Going Down': 4,
    'Sitting': 5,
    'Sitting Down': 6,
    'Standing Up': 7,
    'Standing': 8,
    'Bicycling': 9,
    'Up by Elevator': 10,
    'Down by Elevator': 11,
    'Sitting in Car': 12,
}


class HuGaDB(Dataset):
    '''A dataset class for the HuGaDB dataset (https://github.com/romanchereshnev/HuGaDB).
    Data can be downloaded via Google Drive from the link provided in the repository.
    '''
    # Dataset information.

    LABELS_COL = 5

    HuGaDB_LABELS_IDX = [label for label in HuGaDB_LABELS.keys()]

    NUM_CLASSES = len(HuGaDB_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'gait', 'hugadb')
        else:
            self.root = base_root
        super().__init__()
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.ids = []
        self.x: Dict[str, pd.DataFrame] = {}  # file name -> data
        self.label: Dict[str, Dict[str, List]] = {}  # file name -> label
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        # if no data is present, prompt the user to download it
        if len(os.listdir(self.root)) == 0:
            raise RuntimeError(
                """
                'Visit https://github.com/romanchereshnev/HuGaDB to download the data'
                """
            )
        return self.root

    def build_index(self):
        print('Building index...')
        self.ids = os.listdir(self.root)
        for file in self.ids:
            labels = {}
            file_path = os.path.join(self.root, file)

            # Read the first three lines
            with open(file_path, 'r') as f:
                for i in range(3):
                    line = f.readline().strip()
                    if i < 2:  # Activity, ActivityID
                        key, value = line.split('\t', 1)
                        key = key[1:]  # Remove the '#' from the key
                        value = value.split(' ')
                    else:
                        key, value = line.split(' ', 1)
                        key = key[1:]
                    labels[key] = value

            # Read the rest of the file using pandas
            df = pd.read_csv(file_path, sep='\t', skiprows=3)
            self.x[file] = df
            self.label[file] = labels

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Any:
        filename = self.ids[index]
        data = {
            'x': self.x[filename],
            'label': self.label[filename]
        }

        return data

    @staticmethod
    def num_classes():
        return HuGaDB.NUM_CLASSES
