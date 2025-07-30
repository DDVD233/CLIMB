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
import gdown

BRAIN_TUMOR_LABELS = {
    'no_tumor': 0,
    'glioma_tumor': 1,
    'meningioma_tumor': 2,
    'pituitary_tumor': 3,
}


class BrainTumorDataset(VisionDataset):
    """Dataset class for the Brain Tumor MRI dataset with 4 classes:
    No Tumor, Glioma Tumor, Meningioma Tumor, and Pituitary Tumor
    """

    NUM_CLASSES = 4
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3  # MRI images are typically saved as RGB

    def __init__(self, base_root: str, download: bool = True, train: bool = True) -> None:
        """
        Args:
            base_root: Root directory of the dataset
            train: If True, creates dataset from training set, otherwise from test set
        """
        self.root = os.path.join(base_root, 'mri', 'brain_tumor')
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

        split_folder = 'Training' if self.split == 'train' else 'Testing'
        # Walk through all class directories
        for class_name in self.classes:
            class_dir = os.path.join(self.root, split_folder, class_name)
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
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)

        return index, img.float(), torch.tensor(label).long()

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """
        qa_data = []

        question_prefix = '<image>\nAbove is a brain MRI scan of a patient. '

        for i in range(len(self)):
            img_path = self.images[i]
            label = self.labels[i]

            # Get relative path for storing in json
            rel_path = os.path.relpath(img_path, self.root)

            # Create diagnosis question
            diagnosis_question = ('What type of tumor, if any, is present in this brain MRI scan? '
                                  'Answer with one of the following:\n')
            choices_str = '\n'.join([name.replace('_', ' ').title() for name in self.classes])
            diagnosis_question += choices_str

            # Get diagnosis answer
            diagnosis_answer = self.classes[label].replace('_', ' ').title()

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

        with open(os.path.join(self.root, f'annotation_{self.split}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    @staticmethod
    def num_classes():
        return BrainTumorDataset.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=BrainTumorDataset.INPUT_SIZE,
                patch_size=BrainTumorDataset.PATCH_SIZE,
                in_channels=BrainTumorDataset.IN_CHANNELS
            ),
        ]

    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = dict(zip([self.classes[i] for i in unique], counts))
        return distribution

    def download(self):
        downloader = KaggleDownloader("sartajbhuvaji/brain-tumor-classification-mri")
        downloader.download_file(self.root)
        annotation_ids = [("1vHFqTV35gLlpbtp9u6Yn_9glPXEWg4nk", "annotation_train.jsonl"),
                          ("1HPdKj1U4JIQiElpK4qL9MiflQdUYnXSd", "annotation_test.jsonl")
                          ]
        for a_id, a_name in annotation_ids:
            gdown.download(f"https://drive.google.com/uc?id={a_id}",
                           os.path.join(self.root, a_name), quiet=False)

        print("Successfully downloaded Brain Tumor dataset")


if __name__ == "__main__":
    d = BrainTumorDataset(download=True, base_root='data')