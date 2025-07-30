import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

from src.datasets.specs import Input2dSpec
import gdown

TRAIN_SPLIT_RATIO = 0.8


def any_exist(files):
    return any(map(os.path.exists, files))


class pad_ufes_20(Dataset):
    # Dataset information.
    ''' 
    A dataset class for the PAD_UEFS_20 Dermatology dataset. (https://data.mendeley.com/datasets/zr7vgbcyr2/1)
    
    The PAD-UFES-20 dataset was collected along with the Dermatological and Surgical Assistance Program at the Federal University of Espírito Santo (UFES-Brazil), which is     a nonprofit program that provides free skin lesion treatment, in particular, to low-income people who cannot afford private treatment.
    The dataset consists of 2,298 samples of six different types of skin lesions. Each sample consists of a clinical image and up to 22 clinical features including the         patient's age, skin lesion location, Fitzpatrick skin type, and skin lesion diameter.  The skin lesions are: Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC),     Actinic Keratosis (ACK), Seborrheic Keratosis (SEK), Bowen’s disease (BOD), Melanoma (MEL), and Nevus (NEV). As the Bowen’s disease is considered SCC in situ, we           clustered them together, which results in six skin lesions in the dataset, three skin cancers (BCC, MEL, and SCC) and three skin disease (ACK, NEV, and SEK) 
    
    In total, there are 1,373 patients, 1,641 skin lesions, and 2,298 images present in the dataset.
    
    After download, put your files under a folder called pad_ufes_20, then under a folder called dermatology under your data root.
    '''

    NUM_CLASSES = 5
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'derm', 'pad_ufes_20')
        self.download = download
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.6653, 0.5225, 0.4622], [0.1419, 0.1409, 0.1476])
            ]
        )

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        zip_file = os.path.join(self.root, 'PAD-UFES-20.zip')
        folder = os.path.join(self.root, 'images')
        # if no data is present, prompt the user to download it manually
        if not any_exist([zip_file, folder]):
            if self.download == True:
                self.download_dataset()

            else:
                raise RuntimeError(
                    """
                PAD-UFES-20 data not downloaded,  You can manually download it from https://data.mendeley.com/datasets/zr7vgbcyr2/1
                After download, put your files under a folder called pad_ufes_20, then under a folder called derm under your data root.
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

        filename = "PAD-UEFS-20.zip"
        download_and_extract_archive(
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip",
            download_root=self.root,
            filename=filename
        )

        annotation_ids = [("10_0LvCPYFmn8A1CO7QYUA-gG4hDTHAlz", "annotation_train.jsonl"),
                          ("1vDr0dDwE_Iuo5PMglf-38aQxlY8WGKie", "annotation_test.jsonl"),
                          ("1teCJSx8DH1BC3feypMdtaBEM0akd2n8-", "annotation_valid.jsonl"),
                          ]
        for a_id, a_name in annotation_ids:
            gdown.download(f"https://drive.google.com/uc?id={a_id}",
                           os.path.join(self.root, a_name), quiet=False)

        print('Done!')

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.root, 'metadata.csv')
        df = pd.read_csv(index_file)
        df['image_name'] = df['img_id'].apply(lambda s: os.path.join(self.root, 'images/' + s))
        df = pd.concat([df, pd.get_dummies(df["diagnostic"])], axis=1)
        # BCC, SCC, ACK, MEL, NEV, Other

        #rename sek into other, to help unify our ML task formulation
        df = df.rename({'SEK': 'OTHER'}, axis=1)
        #merge ack and scc into one class akiec, as suggested by Dr. Adamson, since AK and SCC only have size differences
        df['AKIEC'] = np.where((df['ACK'] == 1) | (df['SCC'] == 1), 1, 0)
        df = df.drop(columns=['ACK', 'SCC'])
        # Split into train/val
        cols = ['MEL', 'NEV', 'BCC', 'AKIEC', 'OTHER']
        df = df.fillna(0)
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
        self.fnames = df['image_name'].to_numpy()
        self.labels = df[['MEL', 'NEV', 'BCC', 'AKIEC', 'OTHER']].to_numpy()
        print('Done')

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> Any:
        fname = self.fnames[index]
        image = Image.open(fname).convert('RGB')
        img = self.TRANSFORMS(image)
        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
            # print(dim_gap)
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

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        Each sample will contain a clinical image and a diagnostic question with the corresponding answer.
        """

        qa_data = []
        question_prefix = '<image>\nAbove is a clinical image of a patient. '
        labels = ['Melanoma', 'Nevus', 'Basal Cell Carcinoma', 'AKIEC', 'Other']
        idx_to_label = {i: label for i, label in enumerate(labels)}

        for i in range(len(self)):
            label = self.labels[i]
            file_name = self.fnames[i]
            relative_path = os.path.relpath(file_name, self.root)

            # Create a question for the diagnosis
            diagnosis_question = 'What is the diagnosis of the patient in the clinical image? Answer with one word from the following:'
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

        # Save the QA data to a JSONL file
        with open(os.path.join(self.root, f'annotation_{self.split}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    @staticmethod
    def num_classes():
        return pad_ufes_20.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=pad_ufes_20.INPUT_SIZE, patch_size=pad_ufes_20.PATCH_SIZE, in_channels=pad_ufes_20.IN_CHANNELS
            ),
        ]
