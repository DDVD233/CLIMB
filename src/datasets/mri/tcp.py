import os
from typing import Any, List, Optional

import nibabel as nib
import numpy as np
import pandas
from pandas import read_csv
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec

# From DATASET_ROOT/chexpert/CheXpert-v1.0-small/valid.csv
TRANSDIAGNOSTIC_LABELS = {
    'No Depression': 0,
    'Mild Depression': 1,
    'Moderate Depression': 2,
    'Severe Depression': 3,
    'Extremely Severe Depression': 4,
    'No Anxiety': 5,
    'Mild Anxiety': 6,
    'Moderate Anxiety': 7,
    'Severe Anxiety': 8,
    'Extremely Severe Anxiety': 9,
    'No Stress': 10,
    'Mild Stress': 11,
    'Moderate Stress': 12,
    'Severe Stress': 13,
    'Extremely Severe Stress': 14,
}


def any_exist(files):
    return any(map(os.path.exists, files))


def map_depression(score):
    if score == 999:
        return 'Unknown'
    elif score >= 28:
        return 'Extremely Severe Depression'
    elif score >= 21:
        return 'Severe Depression'
    elif score >= 14:
        return 'Moderate Depression'
    elif score >= 10:
        return 'Mild Depression'
    else:
        return 'No Depression'

def map_anxiety(score):
    if score == 999:
        return 'Unknown'
    elif score >= 20:
        return 'Extremely Severe Anxiety'
    elif score >= 15:
        return 'Severe Anxiety'
    elif score >= 10:
        return 'Moderate Anxiety'
    elif score >= 8:
        return 'Mild Anxiety'
    else:
        return 'No Anxiety'

def map_stress(score):
    if score == 999:
        return 'Unknown'
    elif score >= 34:
        return 'Extremely Severe Stress'
    elif score >= 26:
        return 'Severe Stress'
    elif score >= 19:
        return 'Moderate Stress'
    elif score >= 15:
        return 'Mild Stress'
    else:
        return 'No Stress'


def map_subtype(mri_type) -> List[str]:
    if mri_type == 'anat':
        return ['run-01_T1w', 'run-01_T2w']
    elif mri_type == 'fmap':
        return ['dir-ap_epi', 'dir-pa_epi']
    elif mri_type == 'func':
        return ['task-restAP_run-01_bold', 'task-restAP_run-02_bold', 'task-restPA_run-01_bold']


def drop_underscore(s):
    return s.replace('_', '')


class TCP(Dataset):
    '''A dataset class for the Transdiagnostic Connectome dataset (https://openneuro.org/datasets/ds005237/versions/1.0.3/download).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.

    LABELS_COL = 5

    TRANSDIAGNOSTIC_LABELS_IDX = [label for label in TRANSDIAGNOSTIC_LABELS.keys()]

    NUM_CLASSES = len(TRANSDIAGNOSTIC_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 mri_types: List[str]=['anat', 'fmap', 'func'], in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'mri', 'tcp')
        else:
            self.root = base_root
        super().__init__()
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.data: Optional[pandas.DataFrame] = None
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.5035], [0.2883])
            ]
        )
        self.mri_types = mri_types

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        annotation_path = os.path.join(self.root, 'phenotype')
        # if no data is present, prompt the user to download it
        if not os.path.exists(annotation_path):
            raise RuntimeError(
                """
                'Visit https://openneuro.org/datasets/ds005237/versions/1.0.3/download to download the data'
                """
            )

        # return the data folder
        annotation_file = 'dass01.csv'
        assert os.path.exists(os.path.join(annotation_path, annotation_file)), \
            'Annotation File \"dass01.csv\" not found'
        return annotation_path

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.index_location, 'dass01.csv')
        # Read csv
        df = read_csv(index_file, skiprows=1)
        df['depression_label'] = df['dass_depr_sc'].apply(map_depression)
        df['anxiety_label'] = df['dass_anx_sc'].apply(map_anxiety)
        df['stress_label'] = df['dass_stress_sc'].apply(map_stress)
        df['subjectkey'] = df['subjectkey'].apply(drop_underscore)

        # Filter out folders with missing MRI data
        for data in df.iterrows():
            folder_name = f"sub-{data[1]['subjectkey']}"
            if not os.path.exists(os.path.join(self.root, folder_name)):
                df.drop(data[0], inplace=True)
                print(f"Folder {folder_name} not found. Dropping row.")

        self.data = df

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        data = self.data.iloc[index].to_dict()
        folder_name = f"sub-{data['subjectkey']}"

        for mri_type in self.mri_types:
            sub_types = map_subtype(mri_type)
            for sub_type in sub_types:
                mri_path = os.path.join(self.root, folder_name, mri_type, f'{folder_name}_{sub_type}.nii.gz')
                mri_image = nib.load(mri_path)
                img = mri_image.get_fdata()
                data[f"{mri_type}_{sub_type}"] = img

        return data

    @staticmethod
    def num_classes():
        return TCP.NUM_CLASSES