import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from torch.utils.data import Dataset

from src.load_video import load_video


class EchoNetDynamic(Dataset):
    '''A dataset class for the EchoNet Dynamic dataset (https://aimi.stanford.edu/echonet-dynamic-cardiac-ultrasound).
    '''
    # Dataset information.
    INPUT_SIZE = (112, 112)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'ultrasound', 'EchoNet-Dynamic')
        else:
            self.root = base_root
        super().__init__()
        self.index_location = self.find_data()
        self.split: str = 'train' if train else 'valid'
        self.file_list: Optional[pd.DataFrame] = None
        self.file_names: List[str] = []
        self.label: Optional[pd.DataFrame] = None
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        annotation_path = os.path.join(self.root, 'VolumeTracings.csv')
        # if no data is present, prompt the user to download it
        if not os.path.exists(annotation_path):
            raise RuntimeError(
                """
                'Visit https://aimi.stanford.edu/echonet-dynamic-cardiac-ultrasound to download the data'
                """
            )
        return annotation_path

    def build_index(self):
        print('Building index...')
        file_list_path = os.path.join(self.root, 'FileList.csv')
        file_list = pd.read_csv(file_list_path)
        # Filter split
        file_list = file_list[file_list['Split'] == self.split.upper()]
        self.file_list = file_list
        self.file_names = file_list['FileName'].tolist()
        label_path = os.path.join(self.root, 'VolumeTracings.csv')
        self.label = pd.read_csv(label_path)

    def __len__(self) -> int:
        return self.label.shape[0]

    def __getitem__(self, index: int) -> Any:
        file_name = self.file_names[index]
        file_info = self.file_list.iloc[index]

        volumes = self.label[self.label['FileName'] == f'{file_name}.avi']

        # load video to frames
        file_path = f'Videos/{file_name}.avi'
        frames = load_video(os.path.join(self.root, file_path))

        data = {
            'file_path': f'Videos/{file_name}.avi',
            'volumes': volumes,
            'ef': file_info['EF'],
            'edv': file_info['EDV'],
            'esv': file_info['ESV'],
            'height': file_info['FrameHeight'],
            'width': file_info['FrameWidth'],
            'fps': file_info['FPS'],
            'frames': frames
        }

        return data
