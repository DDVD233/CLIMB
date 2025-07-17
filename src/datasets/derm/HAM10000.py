import json
import os
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
import gdown

from src.datasets.specs import Input2dSpec

TRAIN_SPLIT_RATIO = 0.8


def any_exist(files):
    return any(map(os.path.exists, files))


class HAM10000(Dataset):
    # Dataset information.
    """
    The HAM10000 ("Human Against Machine with 10000 training images") dataset. Dermatoscopic images were collected from different populations, acquired and stored by   different modalities. The final dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).
    
    After download, put your files under a folder called ham10000, then under a folder called dermatology under your data root.
    """
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3
    NUM_CLASSES = 5

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'derm', 'ham10000')
        self.download = download
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.7635, 0.5462, 0.5705], [0.1404, 0.1520, 0.1693])
            ]
        )

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        zip_file = os.path.join(self.root, 'ISIC2018_Task3_Training_Input.zip')
        folder = os.path.join(self.root, 'ISIC2018_Task3_Training_Input')
        # if no data is present, prompt the user to download it
        if not any_exist([zip_file, folder]):
            if self.download == True:
                self.download_dataset()

            else:
                raise RuntimeError(
                    """
                HAM10000 data not downloaded,  You can use download=True to download it
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
        print('Downloading and Extracting...')

        filename = "ISIC2018_Task3_Training_Input.zip"
        download_and_extract_archive(
            "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip",
            download_root=self.root,
            filename=filename
        )

        annotation_ids = [("19LGZu80eTPrESwTPNLh63wcHUbnq354C", "annotation_train.jsonl"),
                          ("1YiNNMrd4DjMNYCH3hIyqdNCaTiM9t8sz", "annotation_valid.jsonl")
                          ]
        for a_id, a_name in annotation_ids:
            gdown.download(f"https://drive.google.com/uc?id={a_id}",
                           os.path.join(self.root, a_name), quiet=False)

        print('Done!')

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.root, 'ISIC2018_Task3_Training_GroundTruth.csv')
        df = pd.read_csv(index_file)
        #MEL    NV	BCC	AKIEC	BKL	DF	VASC
        df['image'] = df['image'].apply(lambda s: os.path.join(self.root, 'ISIC2018_Task3_Training_Input', s + '.jpg'))
        #merge bkl df vasc into other, since they are relatively less frequent classes, also helps unify our ML task formulation
        df['OTHER'] = np.where((df['BKL'] == 1) | (df['DF'] == 1) | (df['VASC'] == 1), 1, 0)
        df = df.drop(columns=['BKL', 'DF', 'VASC'])
        # Split into train/val
        cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'OTHER']
        index_file = df.fillna(0)

        for c in cols:
            df.loc[(df[c] < 0), c] = 0
        index = pd.DataFrame(columns=df.columns)
        df['labels'] = df[cols].idxmax(axis=1)
        index_file = df.copy()
        cols = ['labels']
        index = pd.DataFrame(columns=index_file.columns)
        for c in cols:
            unique_counts = index_file[c].value_counts()
            for c_value, _ in unique_counts.items():
                df_sub = index_file[index_file[c] == c_value]
                g = df_sub.sample(frac=TRAIN_SPLIT_RATIO, replace=False)
                # index = index.append(g)
                index = pd.concat([index, g])
        index_file = index.reset_index(drop=True)
        if self.split != 'train':
            index_file = pd.concat([df, index_file]).drop_duplicates(keep=False)
        df = index_file.reset_index(drop=True)
        self.labels = df[['MEL', 'NV', 'BCC', 'AKIEC', 'OTHER']].to_numpy()
        self.fnames = df['image'].to_numpy()
        print('Done')

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> Any:
        fname = self.fnames[index]
        image = Image.open(os.path.join(self.root, fname)).convert('RGB')
        img = self.TRANSFORMS(image)
        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
        elif h == w:
            #edge case 223,223, resize to match size 224*224
            dim_gap = self.INPUT_SIZE[0] - h
            pad1, pad2 = dim_gap, dim_gap
            img = transforms.Pad((pad1, pad2, 0, 0))(img)
        else:
            dim_gap = img.shape[2] - img.shape[1]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((0, pad1, 0, pad2))(img)
        label = np.argmax(self.labels[index])
        return index, img.float(), label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """

        qa_data = []
        question_prefix = '<image>\nAbove is a dermoscopy image of a patient. '
        labels = ['Melanoma', 'Nevus', 'Basal Cell Carcinoma', 'AKIEC', 'Other']
        idx_to_label = {i: label for i, label in enumerate(labels)}

        for i in range(len(self)):
            label = self.labels[i]
            file_name = self.fnames[i]
            relative_path = os.path.relpath(file_name, self.root)

            # Create a question for each label
            diagnosis_question = 'What is the diagnosis of the patient in the dermoscopy image? Answer with one word from the following:'
            choices_str = '\n'.join(labels)
            diagnosis_question += f'\n{choices_str}'
            diagnosis_strings = [f'{idx_to_label[j]}' for j in range(len(label)) if label[j] == 1]

            if len(diagnosis_strings) == 0:
                print(f'No diagnosis for {relative_path}')
                continue

            diagnosis_answer = ', '.join(diagnosis_strings)
            diagnosis_annotation = {
                'images': [relative_path],
                'explanation': '',
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
        return HAM10000.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=HAM10000.INPUT_SIZE, patch_size=HAM10000.PATCH_SIZE, in_channels=HAM10000.IN_CHANNELS),
        ]
