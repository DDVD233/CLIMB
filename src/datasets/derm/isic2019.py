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


class ISIC2019(Dataset):
    # Dataset information.
    """
    ISIC2019 Dataset has a goal of classifying dermoscopic images among nine different diagnostic categories. 25,331 images are available for training across 8 different categories. We transformed them into 5 classes. 
    
    After download, put your files under a folder called isic2019, then under a folder called dermatology under your data root.
    """
    LABEL_FRACS = {'small': 8, 'medium': 64, 'large': 256, 'full': np.inf}

    # mean: tensor([0.6678, 0.5298, 0.5245])
    # std:  tensor([0.2231, 0.2029, 0.2145])
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3
    NUM_CLASSES = 5
    CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'OTHER']
    LABEL_FRACS = {'small': 8, 'medium': 64, 'large': 256, 'full': np.inf}

    def __init__(self, base_root: str, download: bool = False, train: bool = True, finetune_size: str = None) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'derm', 'isic2019')

        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'

        self.finetune_size = 0 if finetune_size is None else ISIC2019.LABEL_FRACS[finetune_size]
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.6678, 0.5298, 0.5245], [0.2231, 0.2029, 0.2145])
            ]
        )

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        zip_file = os.path.join(self.root, 'ISIC_2019_Training_Input.zip')
        folder = os.path.join(self.root, 'ISIC_2019_Training_Input')
        # if no data is present, prompt the user to download it manually from https://challenge.isic-archive.com/landing/2019/
        if not any_exist([zip_file, folder]):
            raise RuntimeError(
                """
                ISIC 2019 data not downloaded,  get it from https://challenge.isic-archive.com/landing/2019/ manually
                After download, put your files under a folder called isic2019, then under a folder called dermatology under your data root.
  
                """
            )

        # if the data has not been extracted, extract the data
        if not os.path.exists(folder):
            print('Extracting data...')
            extract_archive(zip_file)
            print('Done')

        # return the data folder
        return folder

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.root, 'ISIC_2019_Training_GroundTruth.csv')
        df = pd.read_csv(index_file)
        df['image_name'] = df['image'].apply(lambda s: os.path.join(self.root, 'ISIC_2019_Training_Input/' + s + '.jpg'))
        #merge bkl, df, vasc into other, since they are less frequent classes, also helps unify our ML task formulation
        df['OTHER'] = np.where((df['BKL'] == 1) | (df['DF'] == 1) | (df['VASC'] == 1), 1, 0)
        #merge ack and scc into one class akiec, as suggested by Dr. Adamson, since AK and SCC only have size differences
        df['AKIEC'] = np.where((df['AK'] == 1) | (df['SCC'] == 1), 1, 0)
        df = df.drop(columns=['BKL', 'DF', 'VASC', 'UNK', 'AK', 'SCC'])

        cols = ISIC2019.CLASSES

        for c in cols:
            df.loc[(df[c] < 0), c] = 0
        index = pd.DataFrame(columns=df.columns)
        df['labels'] = df[cols].idxmax(axis=1)
        index_file = df.copy()
        cols = ['labels']
        # if finetuning, get 'finetune_size' labels for each class
        # if insufficient examples, use all examples from that class
        for c in cols:
            unique_counts = index_file[c].value_counts()
            for c_value, l in unique_counts.items():
                df_sub = index_file[index_file[c] == c_value]
                if self.finetune_size > 0:
                    #if finetune
                    g = df_sub.sample(n=min(l, self.finetune_size), replace=False)
                else:
                    #if train
                    g = df_sub.sample(frac=TRAIN_SPLIT_RATIO, replace=False)
                index = index.append(g)
        index_file = index.reset_index(drop=True)
        #if valid
        if self.split != 'train':
            index_file = pd.concat([df, index_file]).drop_duplicates(keep=False)
        df = index_file.reset_index(drop=True)
        self.fnames = df['image_name'].to_numpy()
        #5 classes defined to generalize across dermatology tasks
        self.labels = df[['MEL', 'NV', 'BCC', 'AKIEC', 'OTHER']].values
        print('Done')

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> Any:
        fname = self.fnames[index]

        img = Image.open(fname).convert('RGB')
        img = self.TRANSFORMS(img)
        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
        elif h == w:
            #edge case 223,223,  resize to match 224*224
            dim_gap = self.INPUT_SIZE[0] - h
            pad1, pad2 = dim_gap, dim_gap
            img = transforms.Pad((pad1, pad2, 0, 0))(img)
        else:
            dim_gap = img.shape[2] - img.shape[1]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((0, pad1, 0, pad2))(img)

        label = torch.tensor(np.argmax(self.labels[index])).item()
        return index, img.float(), label

    @staticmethod
    def num_classes():
        return ISIC2019.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=ISIC2019.INPUT_SIZE, patch_size=ISIC2019.PATCH_SIZE, in_channels=ISIC2019.IN_CHANNELS),
        ]
