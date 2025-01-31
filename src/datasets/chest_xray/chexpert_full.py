import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec

# From DATASET_ROOT/chexpert/CheXpert-v1.0-small/valid.csv
CHEXPERT_LABELS = {
    'No Finding': 0,
    'Enlarged Cardiomediastinum': 1,
    'Cardiomegaly': 2,
    'Lung Opacity': 3,
    'Lung Lesion': 4,
    'Edema': 5,
    'Consolidation': 6,
    'Pneumonia': 7,
    'Atelectasis': 8,
    'Pneumothorax': 9,
    'Pleural Effusion': 10,
    'Pleural Other': 11,
    'Fracture': 12,
    'Support Devices': 13,
}


def any_exist(files):
    return any(map(os.path.exists, files))


class CheXpert_Full(VisionDataset):
    '''A dataset class for the CheXpert dataset (https://stanfordmlgroup.github.io/competitions/chexpert/).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.

    LABELS_COL = 5

    # index to unified chexpert format
    CHEXPERT_LABELS_IDX = np.array(
        [
            CHEXPERT_LABELS['Atelectasis'], CHEXPERT_LABELS['Enlarged Cardiomediastinum'], CHEXPERT_LABELS['Cardiomegaly'],
            CHEXPERT_LABELS['Lung Opacity'], CHEXPERT_LABELS['Lung Lesion'], CHEXPERT_LABELS['Edema'],
            CHEXPERT_LABELS['Consolidation'], CHEXPERT_LABELS['Pneumonia'], CHEXPERT_LABELS['Atelectasis'],
            CHEXPERT_LABELS['Pneumothorax'], CHEXPERT_LABELS['Pleural Effusion'], CHEXPERT_LABELS['Pleural Other'],
            CHEXPERT_LABELS['Fracture'], CHEXPERT_LABELS['Support Devices']
        ],
        dtype=np.int32
    )
    IDX_TO_LABEL = {v: k for k, v in CHEXPERT_LABELS.items()}

    NUM_CLASSES = 14  # 14 total: len(self.CHEXPERT_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'chest_xray', 'chexpert_full')
        super().__init__(self.root)
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.5035], [0.2883])
            ]
        )

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        components = list(map(lambda x: os.path.join(self.root, 'CheXpert-v1.0' + x), ['', '.zip']))
        # if no data is present, prompt the user to download it
        if not any_exist(components):
            raise RuntimeError(
                """
                'Visit https://stanfordmlgroup.github.io/competitions/chexpert/ to download the data'
                'Once you receive the download links, place the zip file in {}'.format(self.root)
                'To maintain compatibility with the paper baselines, download the full version, if you want to use a smaller version, change to experiment on chexpert   instead of chexpert_full in your configs.'
                """
            )

        # if the data has not been extracted, extract the data, prioritizing the full-res dataset
        if not any_exist(components[:1]):
            if os.path.exists(components[1]):
                print('Extracting data...')
                extract_archive(components[1])
                print('Done')

        # return the data folder
        for i in (0, 1):
            if os.path.exists(components[i]):
                return components[i]
        raise FileNotFoundError('CheXpert data (full) not found')

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.index_location, self.split + '.csv')
        self.fnames = np.loadtxt(index_file, dtype=str, delimiter=',', skiprows=1, usecols=0)

        end_col = self.LABELS_COL + len(CHEXPERT_LABELS)
        # missing values occur when no comment is made on a particular diagnosis. we treat this as a negative diagnosis
        self.labels = np.genfromtxt(
            index_file,
            dtype=float,
            delimiter=',',
            skip_header=1,
            usecols=range(self.LABELS_COL, end_col),
            missing_values='',
            filling_values=0,
        )
        self.labels = np.maximum(self.labels, 0)  # convert -1 (unknown) to 0
        print('Done')

    def __len__(self) -> int:
        return self.fnames.shape[0]

    def __getitem__(self, index: int) -> Any:
        fname = self.fnames[index]
        image = Image.open(os.path.join(self.root, fname)).convert("L")
        img = self.TRANSFORMS(image)
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
        label = torch.tensor(self.labels[index][self.CHEXPERT_LABELS_IDX]).long()
        return index, img.float(), label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """

        qa_data = []

        question_prefix = '<image>\nAbove is a chest X-ray image of a patient. '

        for i in range(len(self)):
            label: np.ndarray = self.labels[i]
            fname = self.fnames[i]

            # Create a question for each label
            diagnosis_question = ('What is the diagnosis of the patient in the X-ray image?'
                                  'Answer with one or multiple phrases from the following:')
            choices_str = '\n'.join(CHEXPERT_LABELS.keys())
            diagnosis_question += f'\n{choices_str}'
            diagnosis_strings = [f'{self.IDX_TO_LABEL[j]}' for j in range(len(label)) if label[j] == 1]

            if len(diagnosis_strings) == 0:
                print(f'No diagnosis for {fname}')
                continue

            diagnosis_answer = ', '.join(diagnosis_strings)
            diagnosis_annotation = {
                'images': [fname],
                'explanation': '',
                'label': label[self.CHEXPERT_LABELS_IDX].tolist(),
                'label_type': 'chexpert',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': diagnosis_answer}
                ]
            }

            qa_data.append(diagnosis_annotation)

        with open(os.path.join(self.root, f'annotation_{self.split}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data


    @staticmethod
    def num_classes():
        return CheXpert_Full.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=CheXpert_Full.INPUT_SIZE,
                patch_size=CheXpert_Full.PATCH_SIZE,
                in_channels=CheXpert_Full.IN_CHANNELS
            ),
        ]