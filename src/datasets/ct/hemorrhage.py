import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec

HEMORRHAGE_LABELS = {
    'No_Hemorrhage': 0,
    'Has_Hemorrhage': 1,
}


class BrainCTHemorrhageDataset(VisionDataset):
    """Dataset class for Brain CT Images with binary hemorrhage classification
    Classes: No Hemorrhage (0) vs Has Hemorrhage (1)
    """

    NUM_CLASSES = 2
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1  # CT scans are grayscale

    def __init__(self, base_root: str, train: bool = True, test_size: float = 0.2,
                 random_state: int = 42) -> None:
        """
        Args:
            base_root: Root directory of the dataset
            train: If True, creates dataset from training set, otherwise from test set
            test_size: Proportion of the dataset to include in the test split
            random_state: Random state for train-test split
        """
        self.root = os.path.join(base_root, 'ct', 'hemorrhage')
        super().__init__(self.root)

        self.split = 'train' if train else 'test'
        self.classes = list(HEMORRHAGE_LABELS.keys())
        self.class_to_idx = HEMORRHAGE_LABELS
        self.test_size = test_size
        self.random_state = random_state

        self.build_index()

        self.transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Standard normalization for medical images
        ])

    def build_index(self):
        """Build index of all images and their corresponding labels from the CSV file"""
        # Read the diagnosis CSV file
        diagnosis_df = pd.read_csv(os.path.join(self.root, 'hemorrhage_diagnosis.csv'))



        # Create image paths and get labels
        image_paths = []
        labels = []

        for _, row in diagnosis_df.iterrows():
            # Construct image path
            patient_num = str(row['PatientNumber']).zfill(3)
            slice_num = str(row['SliceNumber'])
            img_path_1 = os.path.join(self.root, 'Patients_CT',
                                    f'{patient_num}', 'brain',
                                    f'{slice_num}.jpg')
            img_path_2 = os.path.join(self.root, 'Patients_CT',
                                    f'{patient_num}', 'bone',
                                    f'{slice_num}.jpg')
            assert os.path.exists(img_path_1) or os.path.exists(img_path_2), f'Image not found: {img_path_1} or {img_path_2}'
            images = []
            if os.path.exists(img_path_1):
                images.append(img_path_1)
            else:
                images.append(img_path_2)
            if os.path.exists(img_path_2):
                images.append(img_path_2)
            else:
                images.append(img_path_1)
            image_paths.append(images)
            # Convert No_Hemorrhage to binary classification
            # 0 for No_Hemorrhage (1 in CSV), 1 for Has_Hemorrhage (0 in CSV)
            labels.append(0 if row['No_Hemorrhage'] == 1 else 1)

        # Convert to numpy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # Perform train-test split while maintaining patient-level separation
        patient_nums = np.array([int(path[0].split('Patients_CT/')[1][:3])
                                 for path in image_paths])
        unique_patients = np.unique(patient_nums)

        # Split patients into train and test
        train_patients, test_patients = train_test_split(
            unique_patients,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Select images based on split
        if self.split == 'train':
            mask = np.isin(patient_nums, train_patients)
        else:
            mask = np.isin(patient_nums, test_patients)

        self.images = image_paths[mask]
        self.labels = labels[mask]

        # Store patient numbers for reference
        self.patient_nums = patient_nums[mask]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (index, image, label) where label is binary (0: No Hemorrhage, 1: Has Hemorrhage)
        """
        img_path = self.images[index]
        label = self.labels[index]

        return index, img_path, label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """
        qa_data = []

        question_prefix = '<image>\n<image>\nAbove is a brain CT scan slice. '

        for i in range(len(self)):
            img_paths = self.images[i]
            label = self.labels[i]

            # Get relative path for storing in json
            rel_paths = [os.path.relpath(img_path, self.root) for img_path in img_paths]

            # Create diagnosis question
            diagnosis_question = ('Is there any hemorrhage present in this CT scan slice? '
                                  'Answer with one of the following:\n')
            choices_str = 'No Hemorrhage\nHas Hemorrhage'
            diagnosis_question += choices_str

            # Get diagnosis answer
            diagnosis_answer = self.classes[label].replace('_', ' ')

            # Create annotation
            diagnosis_annotation = {
                'images': rel_paths,
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': diagnosis_answer}
                ]
            }

            qa_data.append(diagnosis_annotation)

        # Save annotations
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, f'annotation_{self.split}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    @staticmethod
    def num_classes():
        return BrainCTHemorrhageDataset.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=BrainCTHemorrhageDataset.INPUT_SIZE,
                patch_size=BrainCTHemorrhageDataset.PATCH_SIZE,
                in_channels=BrainCTHemorrhageDataset.IN_CHANNELS
            ),
        ]

    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = dict(zip([self.classes[i] for i in unique], counts))
        return distribution

    def get_patient_distribution(self):
        """Get the distribution of slices per patient"""
        unique, counts = np.unique(self.patient_nums, return_counts=True)
        distribution = dict(zip(unique, counts))
        return distribution