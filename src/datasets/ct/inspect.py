import json
import os
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from torchvision.datasets.vision import VisionDataset

from src.datasets.ct.nifti_utils import nifti_to_video
from src.datasets.specs import Input2dSpec

INSPECT_LABELS = {
    'No PE': 0,
    'Acute Subsegmental-only PE': 1,
    'Acute PE': 2,
    'Subsegmental-only PE': 3,
    'Chronic PE': 4
}


class Inspect(VisionDataset):
    """Dataset class for the INSPECT dataset mapped to 5 classes of PE classification"""

    NUM_CLASSES = 5
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        """
        Args:
            base_root: Root directory of the dataset
            train: If True, creates dataset from training set, otherwise from validation set
        """
        self.root = os.path.join(base_root, 'ct', 'inspect')
        super().__init__(self.root)

        self.split = 'train' if train else 'test'
        self.classes = list(INSPECT_LABELS.keys())
        self.class_to_idx = INSPECT_LABELS

        self.build_index()

    def build_index(self):
        """Build index of all images and their corresponding labels"""
        # Read metadata CSV
        metadata_path = os.path.join(self.root, 'Final_Impressions_labels.csv')
        metadata = pd.read_csv(metadata_path)

        # use first 75% of data for training
        if self.split == 'train':
            metadata = metadata.iloc[:int(0.75 * len(metadata))]
        else:
            metadata = metadata.iloc[int(0.75 * len(metadata)):]

        self.labels = []
        self.images = []

        for _, row in tqdm.tqdm(metadata.iterrows(), total=len(metadata)):
            image_id = row['impression_id']
            nii_path = os.path.join(self.root, 'CTPA', f'{image_id}.nii.gz')

            if not os.path.exists(nii_path):
                continue

            # Create videos directory if it doesn't exist
            video_dir = os.path.join(self.root, 'videos')
            os.makedirs(video_dir, exist_ok=True)

            video_path = os.path.join(video_dir, f'{image_id}.mp4')
            if not os.path.exists(video_path):
                try:
                    nifti_to_video(nii_path, video_path)
                except Exception as e:
                    print(f"Error converting {nii_path} to video: {e}")
                    continue

            # Determine PE class
            if row['pe_positive'] == 0:
                label = INSPECT_LABELS['No PE']
            elif row['pe_acute'] == 1 and row['pe_subsegmentalonly'] == 1:
                label = INSPECT_LABELS['Acute Subsegmental-only PE']
            elif row['pe_acute'] == 1:
                label = INSPECT_LABELS['Acute PE']
            elif row['pe_subsegmentalonly'] == 1:
                label = INSPECT_LABELS['Subsegmental-only PE']
            else:
                label = INSPECT_LABELS['Chronic PE']

            self.labels.append(label)
            self.images.append(video_path)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image_path, label) where label is the class index
        """
        img_path = self.images[index]
        label = self.labels[index]

        return img_path, label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """
        qa_data = []
        question_prefix = '<video>\nAbove is a chest CT scan of a patient. '

        for i in range(len(self)):
            img_path = self.images[i]
            label = self.labels[i]

            # Get relative path for storing in json
            rel_path = os.path.relpath(img_path, self.root)

            # Create PE classification question
            classification_question = ('What type of Pulmonary Embolism (PE) is present in this CT scan? '
                                       'Answer with one of the following:\n')
            choices_str = '\n'.join(self.classes)
            classification_question += choices_str

            # Get classification answer
            classification_answer = list(INSPECT_LABELS.keys())[label]

            # Create annotation
            classification_annotation = {
                'videos': [rel_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + classification_question},
                    {'from': 'gpt', 'value': classification_answer}
                ]
            }

            qa_data.append(classification_annotation)

        # Save annotations
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, f'annotation_{self.split.lower()}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    @staticmethod
    def num_classes():
        return Inspect.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=Inspect.INPUT_SIZE,
                patch_size=Inspect.PATCH_SIZE,
                in_channels=Inspect.IN_CHANNELS
            ),
        ]
