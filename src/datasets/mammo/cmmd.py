import json
import os
import shutil

import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.specs import Input2dSpec
from src.utils import count_files, get_pixel_array


class CMMD(Dataset):
    ''' A dataset class for CBIS-DDSM: Breast Cancer Image Dataset.
    (https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
    (https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
    This dataset consists of single breast images, either left or right breast, from one of two views (CC or MLO). 
    Each breast will be categorized on the BIRAD 1-5 scale, which communicates findings on presence/severity of lesions.
    Note: 
        1) Additional preprocessing was used to convert lesion-level BIRAD assessments into breast-level assessments.
        2) You must manually download the data zip from the Kaggle link above into this directory. Rename the the folder you extract from
        the zip file as "cbis". It should contain folders "csv" and "jpeg". 
    '''
    # Dataset information.
    NUM_CLASSES = 2
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'mammo', 'cmmd')
        self.split = 'train' if train else 'test'  # use dataset's test split for validation

        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.build_index()

    def build_index(self):
        print('Building index...')

        annotation_path = os.path.join(self.root, 'CMMD_clinicaldata_revision.xlsx')
        df = pd.read_excel(annotation_path, header=0)

        self.images = []
        self.labels = []

        for i in range(len(df)):
            patient_id = df.iloc[i]['ID1']

            jpg_path = os.path.join(self.root, 'jpegs', patient_id)
            if not os.path.exists(jpg_path):
                dicom_path = os.path.join(self.root, "The-Chinese-Mammography-Database", "\"CMMD\"",
                                          f'"{patient_id}"')
                if not os.path.exists(dicom_path):
                    print(f"Patient {patient_id} not found.")
                    continue
                dicom_path = os.path.join(dicom_path, os.listdir(dicom_path)[0])
                dicom_path = os.path.join(dicom_path, os.listdir(dicom_path)[0])
                dicom_paths = os.listdir(dicom_path)
                for dicom in dicom_paths:
                    this_dicom_path = os.path.join(dicom_path, dicom)
                    dicom_name = dicom.split('.')[0]
                    img_array = get_pixel_array(this_dicom_path)
                    img = Image.fromarray(img_array)
                    if not os.path.exists(jpg_path):
                        os.makedirs(jpg_path)
                    img.save(os.path.join(jpg_path, dicom_name + '.jpg'))

            jpg_paths = [os.path.join(jpg_path, f) for f in os.listdir(jpg_path)]

            self.images.append(jpg_paths)
            self.labels.append(df.iloc[i]['classification'])


        # Make first 75% of data the training set
        split_idx = int(len(self.images) * 0.75)
        if self.split == 'train':
            self.images = self.images[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.images = self.images[split_idx:]
            self.labels = self.labels[split_idx:]

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """

        qa_data = []

        question_prefix = 'Above is a mammography X-ray image of a patient. '

        for i in range(len(self)):
            label: np.ndarray = self.labels[i]
            image_path = self.images[i]
            fnames = [os.path.relpath(f, self.root) for f in image_path]

            image_len = len(image_path)
            this_prefix = '<image>\n' * image_len + question_prefix

            # Create a question for each label
            diagnosis_question = ('Is the abnormality in the image benign or malignant? Answer with one word from'
                                  ' the following:\nBenign\nMalignant')
            assert label in ['Benign', 'Malignant'], f'Invalid label: {label}'
            diagnosis_answer = label

            qa_data.append({
                'images': fnames,
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': this_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': diagnosis_answer}
                ]
            })

        with open(os.path.join(self.root, f'annotation_{self.split}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        return index, img_path, label

    def __len__(self):
        return len(self.images)

    @staticmethod
    def num_classes():
        return CMMD.NUM_CLASSES
