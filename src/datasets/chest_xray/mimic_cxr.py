import getpass
import json
import os
import subprocess
import gzip
import shutil

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

from src.datasets.physionet import PhysioNetDownloader


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

'''
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

    def download_file(self, remote_path, local_path): #, is_folder=False):
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

        #if is_folder:
        #    wget_command = [
        #        'wget',
        #        '-r',  # Recursive download
        #        '-N',  # Only download if newer
        #        '-c',  # Continue partially downloaded files
        #        '--no-check-certificate',  # Skip certificate validation
        #        '--user', self.username,
        #        '--password', self.password,
        #        '-P', local_path,  # Output to specific file
        #        url
        #    ]

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

    #def download_folder(self, remote_path, local_path):
'''
        


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

    def __init__(self, base_root: str, download: bool = True, train: bool = True, finetune_size: str = None, file_list=None):
        self.root = os.path.join(base_root, 'chest_xray', 'mimic-cxr')
        super().__init__(self.root)
        if download:
            if file_list is not None:
                self.file_set = set(file_list)
            else:
                self.file_set = None
            self.download()
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

    def download(self):
        """
        Downloads and extracts the MIMIC-CXR dataset if it is not present.
        """
        # Initialize the downloader
        downloader = PhysioNetDownloader("https://physionet.org/files/mimic-cxr-jpg/2.0.0/")

        # List of essential files required for dataset processing
        required_files = [
            "mimic-cxr-2.0.0-chexpert.csv.gz",
            "mimic-cxr-2.0.0-metadata.csv.gz",
            "mimic-cxr-2.0.0-negbio.csv.gz",
            "mimic-cxr-2.0.0-split.csv.gz",
            "LICENSE.txt",
            "README",
            "SHA256SUMS.txt"
        ]

        # Image data directory (needs to be downloaded manually or automated)
        image_dir = os.path.join(self.root, "files")
        os.makedirs(image_dir, exist_ok=True)

        # Download metadata files
        for file in required_files:
            remote_path = file
            local_path = os.path.join(self.root, file)
            
            if not os.path.exists(local_path):
                print(f"Downloading {file}...")
                success = downloader.download_file(remote_path, local_path)
                if not success:
                    raise RuntimeError(f"Failed to download {file}")

        # Extract the CSV files if needed
        for file in required_files[0:4]:
            file_path = os.path.join(self.root, file)
            extracted_path = file_path.replace(".gz", "")
            if not os.path.exists(extracted_path):
                print(f"Extracting {file}...")
                with gzip.open(file_path, 'rb') as f_in:
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        image_files = []
        metadata = pd.read_csv(os.path.join(self.root, "mimic-cxr-2.0.0-metadata.csv"))

        for i in range(len(metadata['dicom_id'])):
            dicom_id = metadata['dicom_id'][i]
            subj_id = metadata['subject_id'][i]
            study_id = metadata['study_id'][i]
            study_dir = 's' + str(study_id)
            subj_dir = 'p' + str(subj_id)
            if (self.file_set is None or study_dir in self.file_set
                                    or subj_dir in self.file_set
                                    or subj_dir[0:3] in self.file_set
                                    or "files" in self.file_set
                                    or str(dicom_id) in self.file_set):
                image_files.append(os.path.join(subj_dir[0:3], subj_dir, study_dir, str(dicom_id) + '.jpg'))

        #print(image_files)

        # Download image files
        for file in image_files:
            remote_path = os.path.join("files", file)
            local_path = os.path.join(image_dir, file)
            
            if not os.path.exists(local_path):
                print(f"Downloading {file}...")
                success = downloader.download_file(remote_path, local_path)
                if not success:
                    raise RuntimeError(f"Failed to download {file}")







        

        #images_folder = "p10/p10000032/"

        # Download the image dataset
        #if not os.path.exists(os.path.join(image_dir, images_folder)):  # Check if at least one image subdir exists
        #    print("Downloading chest X-ray images...")
        #    #success = downloader.download_file("files/", image_dir)
        #    success = downloader.download_file(os.path.join("files/", images_folder),
        #                                       os.path.join(image_dir, images_folder), is_folder=True)
        #    if not success:
        #        raise RuntimeError("Failed to download chest X-ray images")

        print("Download and extraction completed successfully!")


if __name__ == "__main__":
    d = MIMIC_CXR(base_root='data', file_list=['p10000032', 'p10000764', 's50771383', '8e338050-c72628f4-cf19ef85-cb13d287-5af57beb'])
