import getpass
import json
import os
import subprocess

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from urllib.parse import urljoin

from src.datasets.specs import Input2dSpec


def any_exist(files):
    return any(map(os.path.exists, files))


CHEXPERT_LABELS = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Consolidation': 2,
    'Edema': 3,
    'Enlarged Cardiomediastinum': 4,
    'Fracture': 5,
    'Lung Lesion': 6,
    'Lung Opacity': 7,
    'No Finding': 8,
    'Pleural Effusion': 9,
    'Pleural Other': 10,
    'Pneumonia': 11,
    'Pneumothorax': 12,
    'Support Devices': 13,
}


class PhysioNetDownloader:
    """Handles authentication and downloading of files from PhysioNet using wget"""

    def __init__(self, base_url="https://physionet.org/files/mimic-cxr-jpg/2.0.0/"):
        self.base_url = base_url
        self.credentials_file = os.path.expanduser("~/.physionet_credentials")
        self._load_or_prompt_credentials()

    def _load_or_prompt_credentials(self):
        """Load credentials from file or prompt user"""
        if os.path.exists(self.credentials_file):
            with open(self.credentials_file, 'r') as f:
                self.username = f.readline().strip()
                self.password = f.readline().strip()
        else:
            print("PhysioNet credentials not found. Please enter them now:")
            self.username = input("Username: ")
            self.password = getpass.getpass("Password: ")
            # Save credentials
            os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
            with open(self.credentials_file, 'w') as f:
                f.write(f"{self.username}\n{self.password}")
            os.chmod(self.credentials_file, 0o600)  # Secure the credentials file

    def download_file(self, remote_path, local_path):
        """Download a single file from PhysioNet using wget"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Construct the full URL
        url = urljoin(self.base_url, remote_path)

        # Prepare wget command
        wget_command = [
            'wget',
            '-N',  # Only download if newer
            '-c',  # Continue partially downloaded files
            '--no-check-certificate',  # Skip certificate validation
            '--user', self.username,
            '--password', self.password,
            '-O', local_path,  # Output to specific file
            url
        ]

        try:
            # Run wget command
            result = subprocess.run(
                wget_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            if result.returncode == 0:
                return True
            else:
                print(f"Error downloading {url}: {result.stderr}")
                return False

        except subprocess.SubprocessError as e:
            print(f"Error running wget: {str(e)}")
            return False


class MIMIC_CXR(VisionDataset):
    '''A dataset class for the MIMIC-CXR dataset (https://physionet.org/content/mimic-cxr/2.0.0/).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.
    LABELS = []
    IDX_TO_LABEL = {}
    LABEL_FRACS = {'small': 8, 'medium': 64, 'large': 256, 'full': np.inf}
    NUM_CLASSES = 14  # 14 total: len(self.CHEXPERT_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = True, train: bool = True, finetune_size: str = None):
        self.root = os.path.join(base_root, 'chest_xray', 'mimic-cxr')
        super().__init__(self.root)
        self.index_location = self.find_data()
        self.split = ['train'] if train else ['test', 'validate']
        self.finetune_size = 0 if finetune_size is None else self.LABEL_FRACS[finetune_size]
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.4721], [0.3025])
            ]
        )

        # train mean: tensor([0.4721])
        # train std: tensor([0.3025])

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        components = list(map(lambda x: os.path.join(self.root, 'files' + x), ['', '.zip']))
        # if no data is present, prompt the user to download it
        if not any_exist(components):
            raise RuntimeError(
                """
                'Visit https://physionet.org/content/mimic-cxr-jpg/2.0.0/ to apply for access'
                'Use: wget -r -N -c -np --user [your user name] --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/ to download the data'
                'Once you receive the download links, place the zip file in {}'.format(self.root)'
                'Note cxr-study-list.csv.gz is from https://physionet.org/content/mimic-cxr/2.0.0/
                """
            )

    def build_index(self):
        print('Building index...')
        splits = pd.read_csv(os.path.join(self.root, "mimic-cxr-2.0.0-split.csv.gz"), compression='gzip')
        labels = pd.read_csv(os.path.join(self.root, "mimic-cxr-2.0.0-chexpert.csv.gz"), compression='gzip').fillna(0)
        study_ids = pd.read_csv(os.path.join(self.root, "cxr-study-list.csv.gz"), compression='gzip')
        metadata0 = pd.merge(study_ids, labels, on=['subject_id', 'study_id'])
        metadata = pd.merge(metadata0, splits, on=['subject_id', 'study_id'])
        metadata.to_csv(os.path.join(self.root, "metadata.csv"))
        index_file = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        index_file = index_file[index_file.split.isin(self.split)].reset_index(drop=True)

        index_file['fnames'] = np.array(index_file['path'].apply(lambda x: x.split('.''')[:-1][0])
                                       ) + '/' + np.array(index_file['dicom_id']) + '.jpg'

        # print("checking if all files exist...")
        # index_file = index_file[[os.path.isfile(os.path.join(self.root, i)) for i in index_file['fnames']
        #                         ]].reset_index(drop=True)

        # if finetuning, get 'finetune_size' labels for each class
        # if insufficient examples, use all examples from that class
        if self.split == 'train':
            index_file = index_file.fillna(0)
            cols = list(CHEXPERT_LABELS.keys())
            for c in cols:
                index_file.loc[(index_file[c] < 0), c] = 0

            index_file = index_file.reset_index(drop=True)

        LABELS_COL = index_file.columns.get_loc("Atelectasis")
        end_col = index_file.columns.get_loc("Support Devices") + 1
        self.LABELS = index_file.columns[LABELS_COL:end_col]
        self.IDX_TO_LABEL = {i: self.LABELS[i] for i in range(len(self.LABELS))}
        # missing values occur when no comment is made on a particular diagnosis. we treat this as a negative diagnosis
        self.labels = index_file.iloc[:, range(LABELS_COL, end_col)].values
        self.labels = np.maximum(self.labels, 0)  # convert -1 (unknown) to 0
        self.fnames = index_file['fnames'].values
        print('Done')

    def __len__(self) -> int:
        return self.fnames.shape[0]

    def __getitem__(self, index: int):
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
        label = torch.tensor(self.labels[index]).long()
        return index, img.float(), label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """

        qa_data = []
        downloader = PhysioNetDownloader()
        question_prefix = '<image>\nAbove is a chest X-ray image of a patient. '

        for i in tqdm(range(len(self))):
            label: np.ndarray = self.labels[i]
            fname = self.fnames[i]
            relative_path = fname

            real_path = os.path.join(self.root, fname)
            if not os.path.exists(real_path):
                print(f'File {real_path} does not exist')
                if not downloader.download_file(relative_path, real_path):
                    print(f'Failed to download {fname}')
                    continue
            # try loading the image
            try:
                image = Image.open(real_path)
            except:
                print(f'Could not load image {fname}')
                if not downloader.download_file(relative_path, real_path):
                    print(f'Failed to download {fname}')
                    continue

            assert os.path.exists(real_path)

            # Create a question for each label
            diagnosis_question = 'What is the diagnosis of the patient in the X-ray image? Answer with one or multiple phrases from the following:'
            choices_str = '\n'.join(CHEXPERT_LABELS.keys())
            diagnosis_question += f'\n{choices_str}'
            diagnosis_strings = [f'{self.IDX_TO_LABEL[j]}' for j in range(len(label)) if label[j] == 1]
            diagnosis_answer = ', '.join(diagnosis_strings)

            if len(diagnosis_strings) == 0:
                # print(f'No diagnosis for {fname}')
                continue

            qa_data.append(
                {
                    'images': [fname],
                    'explanation': '',
                    'conversations': [
                        {'from': 'human', 'value': question_prefix + diagnosis_question},
                        {'from': 'gpt', 'value': diagnosis_answer}
                    ]
                }
            )

        with open(os.path.join(self.root, f'annotation_{self.split[0]}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    @staticmethod
    def num_classes():
        return MIMIC_CXR.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=MIMIC_CXR.INPUT_SIZE, patch_size=MIMIC_CXR.PATCH_SIZE, in_channels=MIMIC_CXR.IN_CHANNELS),
        ]
