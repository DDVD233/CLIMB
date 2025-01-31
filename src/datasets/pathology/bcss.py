import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import tqdm

from src.datasets.specs import Input2dSpec

MAX_SIZE = 1024
MIN_SIZE = 224
OVERLAP = 128  # Overlap between adjacent patches
TEST_SITES = {'OL', 'LL', 'E2', 'EW', 'GM', 'S3'}


class BCSS(Dataset):
    """
    Dataset class for Breast Cancer Semantic Segmentation (BCSS) dataset.
    Stores all annotations in a single JSON file for efficient access.
    """

    NUM_CLASSES = 4  # tumor, stroma, inflammatory, necrosis
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    LABEL_MAP = {
        1: 0,  # tumor -> 0
        2: 1,  # stroma -> 1
        3: 2,  # lymphocytic_infiltrate (inflammatory) -> 2
        4: 3,  # necrosis_or_debris -> 3
    }

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'pathology', 'bcss')
        self.split = 'train' if train else 'test'
        self.jpgs_dir = os.path.join(self.root, 'jpgs')
        self.annotations_file = os.path.join(self.root, 'annotations.json')

        # Create directories if they don't exist
        os.makedirs(self.jpgs_dir, exist_ok=True)

        # Load label mapping
        # self.load_label_mapping()

        # Process images and build index if needed
        self.process_images()
        self.build_index()

        self.TRANSFORMS = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7411, 0.5331, 0.6796],
                                 std=[0.1400, 0.1505, 0.1302])
        ])

    @staticmethod
    def get_site_code(filename: str) -> str:
        """Extract the site code from the image filename."""
        return filename[5:7]

    def process_images(self):
        """Process original images and masks to create classification patches."""
        images_dir = os.path.join(self.root, 'images')
        masks_dir = os.path.join(self.root, 'masks')

        # Skip if already processed
        if os.path.exists(self.annotations_file):
            return

        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        annotations = {'images': {}}

        for img_file in tqdm.tqdm(image_files):
            mask_file = img_file  # Same name for mask and image
            site_code = self.get_site_code(img_file)

            # Load image and mask
            image = cv2.imread(os.path.join(images_dir, img_file))
            mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)

            image_patches = []

            # Process each class
            for original_label, new_label in self.LABEL_MAP.items():
                # Find regions for current class
                regions = self.find_regions(mask, original_label)

                # Extract and save patches for each region
                for i, (x, y, w, h) in enumerate(regions):
                    patch = image[y:y + h, x:x + w]

                    # Calculate overlap information
                    is_split = w == MAX_SIZE or h == MAX_SIZE

                    # Save patch
                    patch_name = f"{img_file[:-4]}_{original_label}_{i}.jpg"
                    cv2.imwrite(os.path.join(self.jpgs_dir, patch_name), patch)

                    # Store patch information
                    patch_info = {
                        'image_name': patch_name,
                        'label': new_label,
                        'original_label': original_label,
                        'bbox': [x, y, w, h],
                        'is_split': is_split,
                        'site_code': site_code
                    }
                    image_patches.append(patch_info)

            # Store all patches for this image
            annotations['images'][img_file] = {
                'site_code': site_code,
                'patches': image_patches
            }

        # Add dataset information
        annotations['dataset_info'] = {
            'num_classes': self.NUM_CLASSES,
            'label_map': self.LABEL_MAP,
            'max_size': MAX_SIZE,
            'min_size': MIN_SIZE,
            'overlap': OVERLAP,
            'test_sites': list(TEST_SITES)
        }

        # Save all annotations
        with open(self.annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)

    def find_regions(self, mask: np.ndarray, label: int) -> List[Tuple[int, int, int, int]]:
        """
        Find bounding boxes of regions with the given label.
        Large regions are split into smaller overlapping patches.
        """
        binary = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Skip regions smaller than minimum size
            if w < MIN_SIZE or h < MIN_SIZE:
                continue

            # If region is larger than maximum size, split it
            if w > MAX_SIZE or h > MAX_SIZE:
                sub_regions = self.split_large_region(x, y, w, h)
                regions.extend(sub_regions)
            else:
                regions.append((x, y, w, h))

        return regions

    def split_large_region(self, x: int, y: int, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Split a large region into smaller overlapping patches."""
        patches = []

        # Calculate number of patches needed in each dimension
        num_w = max(1, int(np.ceil((w - OVERLAP) / (MAX_SIZE - OVERLAP))))
        num_h = max(1, int(np.ceil((h - OVERLAP) / (MAX_SIZE - OVERLAP))))

        # Calculate actual patch size to cover the region evenly
        patch_w = min(MAX_SIZE, w) if num_w == 1 else MAX_SIZE
        patch_h = min(MAX_SIZE, h) if num_h == 1 else MAX_SIZE

        # Calculate steps between patch starts
        step_w = (w - patch_w) / (num_w - 1) if num_w > 1 else 0
        step_h = (h - patch_h) / (num_h - 1) if num_h > 1 else 0

        for i in range(num_h):
            for j in range(num_w):
                # Calculate patch coordinates
                patch_x = x + int(j * step_w)
                patch_y = y + int(i * step_h)

                # Ensure we don't exceed original region boundaries
                patch_x = min(patch_x, x + w - patch_w)
                patch_y = min(patch_y, y + h - patch_h)

                patches.append((patch_x, patch_y, patch_w, patch_h))

        return patches

    def build_index(self):
        """Build index of images and labels for training/test."""
        # Load annotations
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # Flatten patch list
        all_patches = []
        for img_info in self.annotations['images'].values():
            all_patches.extend(img_info['patches'])

        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(all_patches)

        # Split based on site codes
        is_test = df['site_code'].isin(TEST_SITES)

        if self.split == 'test':
            df = df[is_test]
        else:
            # For non-test set, further split into train/val
            df_train = df[~is_test]

            # Random split of non-test sites into train/val
            unique_images = df_train['image_name'].apply(lambda x: x.split('_')[0]).unique()
            np.random.seed(42)
            train_images = np.random.choice(
                unique_images,
                size=int(len(unique_images) * 0.9),  # 90% for training, 10% for validation
                replace=False
            )

            if self.split == 'train':
                df = df_train[df_train['image_name'].apply(lambda x: x.split('_')[0]).isin(train_images)]
            else:  # validation
                df = df_train[~df_train['image_name'].apply(lambda x: x.split('_')[0]).isin(train_images)]

        self.image_names = df['image_name'].tolist()
        self.labels = df['label'].tolist()

        # Print split statistics
        print(f"Split: {self.split}")
        print(f"Number of images: {len(self.image_names)}")
        print("Distribution by site:")
        print(df['site_code'].value_counts())
        print("\nClass distribution:")
        print(df['label'].value_counts())

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_name = self.image_names[index]
        image = Image.open(os.path.join(self.jpgs_dir, image_name)).convert('RGB')
        img = self.TRANSFORMS(image)
        label = self.labels[index]
        return index, img.float(), label

    def to_qa(self):
        """Convert the dataset to a question answering format."""
        qa_data = []
        question_prefix = '<image>\nAbove is a histopathological image patch from breast cancer tissue. '
        labels = ['Tumor', 'Stroma', 'Inflammatory', 'Necrosis']
        idx_to_label = {i: label for i, label in enumerate(labels)}

        for i in range(len(self)):
            image_name = self.image_names[i]
            image_path = os.path.join('jpgs', image_name)
            label = self.labels[i]

            classification_question = 'What type of tissue is shown in this image patch? Choose from:'
            choices_str = '\n'.join(labels)
            classification_question += f'\n{choices_str}'

            classification_answer = idx_to_label[label]

            classification_annotation = {
                'images': [image_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + classification_question},
                    {'from': 'gpt', 'value': classification_answer}
                ]
            }

            qa_data.append(classification_annotation)

        # Save the QA data
        qa_file = os.path.join(self.root, f'annotation_{self.split}.jsonl')
        with open(qa_file, 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    @staticmethod
    def num_classes():
        return BCSS.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=BCSS.INPUT_SIZE,
                patch_size=BCSS.PATCH_SIZE,
                in_channels=BCSS.IN_CHANNELS
            ),
        ]