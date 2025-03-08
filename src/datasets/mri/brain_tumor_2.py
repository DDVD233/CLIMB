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

BRAIN_TUMOR_LABELS = {
    'no': 0,
    'yes': 1,
}

class BinaryBrainTumorDataset(VisionDataset):
    """Dataset class for binary brain tumor classification
    Classes: No Tumor (0) vs Has Tumor (1)
    """

    NUM_CLASSES = 2
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3  # MRI images are typically saved as RGB

    def __init__(self, base_root: str, download: bool = True, train: bool = True) -> None:
        """
        Args:
            base_root: Root directory of the dataset
            train: If True, creates dataset from training set, otherwise from test set
        """
        self.root = os.path.join(base_root, 'mri', 'brain_tumor_2')
        super().__init__(self.root)

        if download:
            self.download()

        self.split = 'train' if train else 'test'
        self.classes = list(BRAIN_TUMOR_LABELS.keys())
        self.class_to_idx = BRAIN_TUMOR_LABELS

        self.build_index()

        self.transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                              std=[0.229, 0.224, 0.225])
        ])

    def build_index(self):
        """Build index of all images and their corresponding labels"""
        self.images = []
        self.labels = []

        # Walk through both class directories
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Get all image files in the class directory
            for root, _, files in os.walk(class_dir):
                for fname in sorted(files):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(root, fname)
                        self.images.append(path)
                        self.labels.append(self.class_to_idx[class_name])

        # Convert to numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        # Create train/test split
        # Using 80% of data for training
        num_samples = len(self.images)
        indices = np.arange(num_samples)
        np.random.seed(42)  # for reproducibility
        np.random.shuffle(indices)
        split_idx = int(0.8 * num_samples)

        if self.split == 'train':
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        self.images = self.images[indices]
        self.labels = self.labels[indices]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (index, image, label) where label is binary (0: No Tumor, 1: Has Tumor)
        """
        img_path = self.images[index]
        label = self.labels[index]

        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)

        return index, img.float(), torch.tensor(label).long()

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """
        qa_data = []

        question_prefix = '<image>\nAbove is a brain MRI scan. '

        for i in range(len(self)):
            img_path = self.images[i]
            label = self.labels[i]

            # Get relative path for storing in json
            rel_path = os.path.relpath(img_path, self.root)

            # Create diagnosis question
            diagnosis_question = ('Is there a tumor present in this brain MRI scan? '
                                'Answer with one of the following:\n')
            choices_str = 'No Tumor\nHas Tumor'
            diagnosis_question += choices_str

            # Get diagnosis answer
            diagnosis_answer = 'Has Tumor' if label == 1 else 'No Tumor'

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
        return BinaryBrainTumorDataset.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=BinaryBrainTumorDataset.INPUT_SIZE,
                patch_size=BinaryBrainTumorDataset.PATCH_SIZE,
                in_channels=BinaryBrainTumorDataset.IN_CHANNELS
            ),
        ]

    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = dict(zip([self.classes[i] for i in unique], counts))
        return distribution

    def download(self):
        downloader = KaggleDownloader("jjprotube/brain-mri-images-for-brain-tumor-detection")
        downloader.download_file(self.root)
        
        



if __name__ == "__main__":
    d = BinaryBrainTumorDataset(download=True, base_root='data')





        
