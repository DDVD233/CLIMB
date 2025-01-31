import json
import os
from typing import Any

import numpy as np
import tqdm
from torchvision.datasets.vision import VisionDataset

from src.datasets.ct.nifti_utils import nifti_to_video
from src.datasets.specs import Input2dSpec

KITS_LABELS = {
    'Benign': 0,
    'Malignant': 1,
}


class Kits23(VisionDataset):
    """Dataset class for the KITS23 dataset mapped to 2 classes: Benign and Malignant
    """

    NUM_CLASSES = 4
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        """
        Args:
            base_root: Root directory of the dataset
            train: If True, creates dataset from training set, otherwise from validation set
        """
        self.root = os.path.join(base_root, 'ct', 'kits23')
        super().__init__(self.root)

        self.split = 'train' if train else 'test'
        self.classes = list(KITS_LABELS.keys())
        self.class_to_idx = KITS_LABELS

        self.build_index()

    def build_index(self):
        """Build index of all images and their corresponding labels"""
        # Read metadata CSV
        metadata_path = os.path.join(self.root, 'dataset', 'kits23.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # use first 75% of data for training
        if self.split == 'train':
            metadata = metadata[:int(0.75 * len(metadata))]
        else:
            metadata = metadata[int(0.75 * len(metadata)):]

        self.labels = []
        self.images = []
        for case in tqdm.tqdm(metadata):
            image_folder = os.path.join(self.root, 'dataset', case['case_id'])
            nii_path = os.path.join(image_folder, 'imaging.nii.gz')
            if not os.path.exists(nii_path):
                continue
            video_path = os.path.join(image_folder, 'imaging.mp4')
            # if not os.path.exists(video_path):
            nifti_to_video(nii_path, video_path)
            if 'malignant' not in case or case['malignant'] is None:
                continue
            label = int(case['malignant'])  # 0 for benign, 1 for malignant
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
            tuple: (index, image, label) where label is the class index
        """
        img_path = self.images[index]
        label = self.labels[index]

        return img_path, label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """
        qa_data = []
        question_prefix = '<video>\nAbove is a kidney CT scan of a patient. '

        for i in range(len(self)):
            img_path = self.images[i]
            label = self.labels[i]

            # Get relative path for storing in json
            rel_path = os.path.relpath(img_path, self.root)

            # Create diagnosis question
            diagnosis_question = ('Is the tumor in the CT Scan benign or malignant? '
                                  'Answer with one of the following:\n')
            choices_str = '\n'.join(self.classes)
            diagnosis_question += choices_str

            # Get diagnosis answer
            diagnosis_answer = list(KITS_LABELS.keys())[label]

            # Create annotation
            diagnosis_annotation = {
                'videos': [rel_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': diagnosis_answer}
                ]
            }

            qa_data.append(diagnosis_annotation)

        # Save annotations
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, f'annotation_{self.split.lower()}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    @staticmethod
    def num_classes():
        return Kits23.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=Kits23.INPUT_SIZE,
                patch_size=Kits23.PATCH_SIZE,
                in_channels=Kits23.IN_CHANNELS
            ),
        ]