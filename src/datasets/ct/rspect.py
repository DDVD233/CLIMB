import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydicom
import tqdm
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec

RSPECT_LABELS = {
    'No PE': 0,
    'Chronic PE': 1,
    'Acute PE': 2
}


def apply_window_level(data, window=700, level=80):
    """
    Apply radiological window/level adjustment optimized for CTPA
    """
    lower = level - window / 2
    upper = level + window / 2
    data_adj = np.clip(data, lower, upper)
    data_adj = ((data_adj - lower) / (window)) * 255
    return data_adj.astype(np.uint8)


def denoise_and_enhance(image):
    """
    Apply denoising and subtle enhancement
    """
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # # Apply bilateral filter for edge-preserving denoising
    # denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply CLAHE for contrast enhancement
    try:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        return enhanced
    except:
        return image


def dicom_to_jpg(dicom_path, output_path):
    """
    Convert a DICOM file to JPG with appropriate windowing for PE visualization
    """
    # Read DICOM file
    dicom = pydicom.dcmread(dicom_path)

    # Get pixel array and convert to float
    image = dicom.pixel_array.astype(float)

    # Rescale to HU units if rescale intercept and slope are present
    if hasattr(dicom, 'RescaleIntercept') and hasattr(dicom, 'RescaleSlope'):
        image = image * dicom.RescaleSlope + dicom.RescaleIntercept

    # Apply windowing
    image = apply_window_level(image)
    image = denoise_and_enhance(image)

    # Save as JPG
    cv2.imwrite(output_path, image)


class Rspect(VisionDataset):
    """Dataset class for the RSPECT dataset mapped to 3 classes: No PE, Chronic PE, Acute PE"""

    NUM_CLASSES = 3
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        """
        Args:
            base_root: Root directory of the dataset
            train: If True, creates dataset from training set, otherwise from validation set
        """
        self.root = os.path.join(base_root, 'ct', 'rspect')
        super().__init__(self.root)

        self.split = 'train' if train else 'test'
        self.classes = list(RSPECT_LABELS.keys())
        self.class_to_idx = RSPECT_LABELS

        self.build_index()

    def build_index(self):
        """Build index of all images and their corresponding labels"""
        # Read metadata CSV
        metadata_path = os.path.join(self.root, f'train.csv')
        metadata = pd.read_csv(metadata_path)

        # Create output directory for JPGs
        jpg_dir = os.path.join(self.root, 'jpgs')
        os.makedirs(jpg_dir, exist_ok=True)

        self.labels = []
        self.images = []

        # Process each instance in the metadata
        for _, row in tqdm.tqdm(metadata.iterrows(), total=len(metadata)):
            study_id = row['StudyInstanceUID']
            series_id = row['SeriesInstanceUID']
            instance_id = row['SOPInstanceUID']

            # if not dcm_path.exists():
            #     continue

            # Create output jpg path
            jpg_name = f"{study_id}_{series_id}_{instance_id}.jpg"
            jpg_path = os.path.join(jpg_dir, jpg_name)

            # Convert DICOM to JPG if not already done
            # if not os.path.exists(jpg_path):
            #     # Find and process the DICOM file
            #     dcm_path = Path(self.root) / "train" / study_id / series_id / f"{instance_id}.dcm"
            #     # assert dcm_path.exists(), f"Missing DICOM file: {dcm_path}"
            #     if not dcm_path.exists():
            #         print(f"Missing DICOM file: {dcm_path}")
            #         continue
            #     try:
            #         dicom_to_jpg(str(dcm_path), jpg_path)
            #     except Exception as e:
            #         print(f"Error processing {dcm_path}: {e}")
            #         continue

            # Determine PE class using instance-level labels
            if row['pe_present_on_image'] == 0:
                label = RSPECT_LABELS['No PE']
            elif row['chronic_pe'] == 1:
                label = RSPECT_LABELS['Chronic PE']
            else:
                label = RSPECT_LABELS['Acute PE']

            self.labels.append(label)
            self.images.append(jpg_path)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        indices = np.random.permutation(len(self.images))
        split_idx = int(0.75 * len(indices))
        if self.split == 'train':
            self.images = self.images[indices[:split_idx]]
            self.labels = self.labels[indices[:split_idx]]
        else:
            self.images = self.images[indices[split_idx:]]
            self.labels = self.labels[indices[split_idx:]]

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
        question_prefix = '<image>\nAbove is a chest CT scan slice of a patient. '

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
            classification_answer = list(RSPECT_LABELS.keys())[label]

            # Create annotation
            classification_annotation = {
                'images': [rel_path],
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
        return Rspect.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=Rspect.INPUT_SIZE,
                patch_size=Rspect.PATCH_SIZE,
                in_channels=Rspect.IN_CHANNELS
            ),
        ]