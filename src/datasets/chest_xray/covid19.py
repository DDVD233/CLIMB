import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec
from src.datasets.kaggle import KaggleDownloader

COVID19_LABELS = {
    'Normal': 0,
    'BacterialPneumonia': 1,
    'ViralPneumonia': 2,
    'COVID-19': 3,
}


class Covid19Dataset(VisionDataset):
    """Dataset class for the Covid19 chest X-ray dataset with 4 classes:
    Normal, Bacterial Pneumonia, Viral Pneumonia, and COVID-19
    """

    NUM_CLASSES = 4
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = True, train: bool = True) -> None:
        """
        Args:
            base_root: Root directory of the dataset
            train: If True, creates dataset from training set, otherwise from validation set
        """
        self.root = os.path.join(base_root, 'chest_xray', 'covid19')
        super().__init__(self.root)

        if download:
            self.download()

        self.split = 'train' if train else 'valid'
        self.classes = list(COVID19_LABELS.keys())
        self.class_to_idx = COVID19_LABELS

        self.build_index()

        self.transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.5035], [0.2883])  # Using same normalization as CheXpert
        ])

    def build_index(self):
        """Build index of all images and their corresponding labels"""
        self.images = []
        self.labels = []

        split_dir = 'TrainData' if self.split == 'train' else 'ValData'
        image_dir = os.path.join(self.root, split_dir)

        # Walk through all class directories
        for class_name in self.classes:
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Get all image files in the class directory
            for root, _, files in os.walk(class_dir):
                for fname in sorted(files):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(root, fname)
                        self.images.append(path)
                        self.labels.append(self.class_to_idx[class_name])

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

        # Load and preprocess image
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        img = self.transform(image)

        # Handle padding similarly to CheXpert
        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
        elif h == w:
            dim_gap = self.INPUT_SIZE[0] - h
            pad1, pad2 = dim_gap, dim_gap
            img = transforms.Pad((pad1, pad2, 0, 0))(img)
        else:
            dim_gap = img.shape[2] - img.shape[1]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((0, pad1, 0, pad2))(img)

        return index, img.float(), torch.tensor(label).long()

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """
        qa_data = []

        question_prefix = '<image>\nAbove is a chest X-ray image of a patient. '

        for i in range(len(self)):
            img_path = self.images[i]
            label = self.labels[i]

            # Get relative path for storing in json
            rel_path = os.path.relpath(img_path, self.root)

            # Create diagnosis question
            diagnosis_question = ('What is the diagnosis of the patient in the X-ray image? '
                                  'Answer with one of the following:\n')
            choices_str = '\n'.join(self.classes)
            diagnosis_question += choices_str

            # Get diagnosis answer
            diagnosis_answer = list(COVID19_LABELS.keys())[label]

            # Create annotation
            diagnosis_annotation = {
                'images': [rel_path],
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
        return Covid19Dataset.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=Covid19Dataset.INPUT_SIZE,
                patch_size=Covid19Dataset.PATCH_SIZE,
                in_channels=Covid19Dataset.IN_CHANNELS
            ),
        ]

    def download(self):
        downloader = KaggleDownloader("darshan1504/covid19-detection-xray-dataset")
        downloader.download_file(self.root)
        
        



if __name__ == "__main__":
    d = Covid19Dataset(download=True, base_root='data')
