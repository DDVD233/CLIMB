import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import extract_archive

from src.datasets.specs import Input2dSpec

TRAIN_SPLIT_RATIO = 0.8


def any_exist(files):
    return any(map(os.path.exists, files))


class ddi(Dataset):
    # Dataset information.
    """
    Diverse Dermatology Images (DDI) dataset—the first publicly available, deeply curated, and pathologically confirmed image dataset with diverse skin tones. The DDI was     retrospectively selected from reviewing pathology reports in Stanford Clinics from 2010-2020.
    
    Please manually register an acccount and download at https://stanfordaimi.azurewebsites.net and put under a folder named ddi, then under a folder called dermatology   under your data root.
    """

    NUM_CLASSES = 2
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'dermatology', 'ddi')
        self.split = 'train' if train else 'valid'
        self.download = download
        self.index_location = self.find_data()
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.5970, 0.4827, 0.4022], [0.2061, 0.2046, 0.2133])
            ]
        )

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        zip_file = os.path.join(self.root, 'ddidiversedermatologyimages.zip')
        folder = os.path.join(self.root, 'ddidiversedermatologyimages')
        # if no data is present, prompt the user to download it
        if not any_exist([zip_file, folder]):
            if self.download == True:
                self.download_dataset()

            else:
                raise RuntimeError(
                    """
                ddi data not downloaded,  You can use download=True to download it
                """
                )

        # if the data has not been extracted, extract the data
        if not os.path.exists(folder):
            print('Extracting data...')
            extract_archive(zip_file)
            print('Done')

        # return the data folder
        return folder

    def download_dataset(self):
        '''Download the dataset if not exists already'''

        # download and extract files
        print('Please register an account and manually download at https://stanfordaimi.azurewebsites.net/')
        print('Done!')

    def build_index(self):
        print('Building index...')
        file_path = os.path.join(self.root, 'ddidiversedermatologyimages', 'ddi_metadata.csv')
        index_file = pd.read_csv(file_path)
        # Split into train/val, by sampling data instances from malignant and non-malignant classes
        cols = ['malignant']
        index = pd.DataFrame(columns=index_file.columns)
        for c in cols:
            unique_counts = index_file[c].value_counts()
            for c_value, _ in unique_counts.items():
                df_sub = index_file[index_file[c] == c_value]
                g = df_sub.sample(frac=TRAIN_SPLIT_RATIO, replace=False)
                index = index.append(g)
        index_file = index.reset_index(drop=True)
        self.fnames = index_file['DDI_file'].to_numpy()
        self.labels = (index_file['malignant'] == True).to_numpy().astype(np.int)
        print('Done')

    def __len__(self) -> int:
        return self.fnames.shape[0]

    def __getitem__(self, index: int) -> Any:
        fname = self.fnames[index]
        image = Image.open(os.path.join(self.root, 'ddidiversedermatologyimages', fname)).convert('RGB')
        img = self.TRANSFORMS(image)
        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
        elif h == w:
            #edge case 223,223, resize to match 224*224
            dim_gap = self.INPUT_SIZE[0] - h
            pad1, pad2 = dim_gap, dim_gap
            img = transforms.Pad((pad1, pad2, 0, 0))(img)
        else:
            dim_gap = img.shape[2] - img.shape[1]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((0, pad1, 0, pad2))(img)
        label = torch.tensor(self.labels[index]).long()
        return index, img.float(), label

    @staticmethod
    def num_classes():
        return ddi.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=ddi.INPUT_SIZE, patch_size=ddi.PATCH_SIZE, in_channels=ddi.IN_CHANNELS),
        ]
