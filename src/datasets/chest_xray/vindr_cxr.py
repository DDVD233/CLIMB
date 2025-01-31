#/home/ubuntu/raw/vindr/physionet.org/files/vindr-cxr/1.0.0
import glob
import json
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

from src.datasets.specs import Input2dSpec
from src.utils import count_files

CHEXPERT_LABELS = {
    'No Finding': 0,
    'Enlarged Cardiomediastinum': 1,
    'Cardiomegaly': 2,
    'Lung Opacity': 3,
    'Lung Lesion': 4,
    'Edema': 5,
    'Consolidation': 6,
    'Pneumonia': 7,
    'Atelectasis': 8,
    'Pneumothorax': 9,
    'Pleural Effusion': 10,
    'Pleural Other': 11,
    'Fracture': 12,
    'Support Devices': 13,
}

mapping = {
    "Aortic enlargement": "Enlarged Cardiomediastinum",
    "Atelectasis": "Atelectasis",
    "Calcification": None,
    "Cardiomegaly": "Cardiomegaly",
    "Clavicle fracture": "Fracture",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Emphysema": None,
    "Enlarged PA": "Enlarged Cardiomediastinum",
    "ILD": None,
    "Infiltration": "Lung Opacity",
    "Lung Opacity": "Lung Opacity",
    "Lung cavity": "Lung Lesion",
    "Lung cyst": "Lung Lesion",
    "Mediastinal shift": "Enlarged Cardiomediastinum",
    "Nodule/Mass": "Lung Lesion",
    "Pleural effusion": "Pleural Effusion",
    "Pleural thickening": "Pleural Other",
    "Pneumothorax": "Pneumothorax",
    "Pulmonary fibrosis": None,
    "Rib fracture": "Fracture",
    "Other lesion": "Lung Lesion",
    "COPD": None,
    "Lung tumor": "Lung Lesion",
    "Pneumonia": "Pneumonia",
    "Tuberculosis": None,
    "Other diseases": None,
    "Other disease": None,
    "No finding": "No Finding",
    "Support Devices": "Support Devices"
}



def any_exist(files):
    return any(map(os.path.exists, files))


class VINDR_CXR(VisionDataset):
    '''A dataset class for the VINDR-CXR dataset (https://physionet.org/content/vindr-cxr/1.0.0/).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.
    LABELS = []
    IDX_TO_LABEL = {}
    NUM_CLASSES = 14  # 14 total: len(self.CHEXPERT_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'chest_xray', 'vindr')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(self.root)
        self.index_location = self.find_data()
        self.split = 'train' if train else 'test'
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.7635], [0.1404])
            ]
        )

    def find_data(self):
        components = list(map(lambda x: os.path.join(self.root, 'train' + x), ['']))
        # if no data is present, prompt the user to download it
        if not any_exist(components):
            raise FileNotFoundError(
                """
                'Visit https://physionet.org/content/vindr-cxr/1.0.0/ to apply for access'
                'Use: wget -r -N -c -np --user [your user name] --ask-password https://physionet.org/files/vindr-cxr/1.0.0 to download the data'
                'Once you receive the download links, download it in {}'.format(self.root)'
                """
            )

        else:
            return components[0]

    def read_dicom(self, file_path: str):
        """Read pixel array from a DICOM file and apply recale and resize
        operations.
        The rescale operation is defined as the following:
            x = x * RescaleSlope + RescaleIntercept
        The rescale slope and intercept can be found in the DICOM files.
        Args:
            file_path (str): Path to a dicom file.
            resize_shape (int): Height and width for resizing.
        Returns:
            The processed pixel array from a DICOM file.
        """

        dcm = pydicom.dcmread(file_path)
        pixel_array = torch.from_numpy(dcm.pixel_array).float().to(self.device)

        # Rescale
        if 'RescaleIntercept' in dcm:
            intercept = dcm.RescaleIntercept
            slope = dcm.RescaleSlope
            pixel_array = pixel_array * slope + intercept

        return pixel_array, dcm

    def apply_windowing(self, image, window_center, window_width):
        """Apply windowing to the image."""
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        image = torch.clamp(image, img_min, img_max)
        return image

    def normalize_to_uint8(self, image):
        """Normalize and convert the image to uint8."""
        image = image - image.min()
        image = image / image.max()
        return (image * 255).byte()

    def dicom_to_jpg(self, fnames, imsize):
        jpeg_dir = os.path.join(self.root, self.split, 'jpegs')
        os.makedirs(jpeg_dir, exist_ok=True)

        for fname in tqdm(fnames):
            dicom_path = fname + ".dicom"
            img_tensor, dcm = self.read_dicom(dicom_path)

            # Try to get window center and width from DICOM tags
            try:
                window_center = dcm.WindowCenter
                window_width = dcm.WindowWidth
                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = window_center[0]
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]
            except AttributeError:
                # If not available, use the image min and max as a fallback
                window_center = (img_tensor.min() + img_tensor.max()) / 2
                window_width = img_tensor.max() - img_tensor.min()

            # Apply windowing
            img_tensor = self.apply_windowing(img_tensor, window_center, window_width)

            # Normalize to uint8
            img_tensor = self.normalize_to_uint8(img_tensor)
            img_tensor = img_tensor.cpu()
            img = transforms.ToPILImage()(img_tensor.squeeze())

            # Save as JPEG
            output_path = os.path.join(jpeg_dir, os.path.basename(fname) + '.jpg')
            img.save(output_path, quality=95)

    def build_index(self):
        print('Building index...')

        metadata = pd.read_csv(os.path.join(self.root, f"annotations/image_labels_{self.split}.csv"))
        index_file = metadata

        dicom_fnames = np.array(index_file['image_id'].apply(lambda x: os.path.join(self.root, f"{self.split}/{x}")))
        if self.split == "train":
            n_files = 15000
        else:
            n_files = 3000
        if (not os.path.isdir(os.path.join(self.root, self.split + "/" + 'jpegs')
                             )) or count_files(os.path.join(self.root, self.split + "/" + 'jpegs')) < n_files:
            if os.path.isdir(os.path.join(self.root, self.split + "/" + 'jpegs')):
                shutil.rmtree(os.path.join(self.root, self.split + "/" + 'jpegs'))
            self.dicom_to_jpg(fnames=dicom_fnames, imsize=self.INPUT_SIZE[0])
        self.fnames = glob.glob(os.path.join(self.root, self.split + "/" + 'jpegs') + "/*.jpg")
        LABELS_COL = index_file.columns.get_loc("Aortic enlargement")
        self.LABELS = index_file.columns[LABELS_COL:]
        self.IDX_TO_LABEL = {i: self.LABELS[i] for i in range(len(self.LABELS))}
        # missing values occur when no comment is made on a particular diagnosis. we treat this as a negative diagnosis
        self.labels = index_file.iloc[:, LABELS_COL:].values
        self.labels = np.maximum(self.labels, 0)  # convert -1 (unknown) to 0
        print('Done')

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> Any:

        fname = self.fnames[index]
        image = Image.open(os.path.join(self.root, fname)).convert("L")
        img = self.TRANSFORMS(image)
        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
        elif h == w:
            #edge case 223,223, resize to match 224*224
            dim_gap = self.INPUT_SIZE[0] - h
            pad1, pad2 = dim_gap, dim_gap
            img = transforms.Pad((pad1, pad2, 0, 0))(img)
        else:
            dim_gap = img.shape[2] - img.shape[1]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((0, pad1, 0, pad2))(img)
        label = torch.tensor(self.labels[index]).long()
        return index, img.float(), label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """

        qa_data = []
        question_prefix = '<image>\nAbove is a chest X-ray image of a patient. '
        chexpert_labels = list(CHEXPERT_LABELS.keys())

        for i in range(len(self)):
            label = self.labels[i]
            file_name = self.fnames[i]
            relative_path = os.path.relpath(file_name, self.root)

            # Create a question for each label
            diagnosis_question = ('What is the diagnosis of the patient in the X-ray image? '
                                  'Answer with one or multiple phrases from the following:')
            choices_str = '\n'.join(CHEXPERT_LABELS.keys())
            diagnosis_question += f'\n{choices_str}'
            diagnosis_strings = [f'{self.IDX_TO_LABEL[j]}' for j in range(len(label)) if label[j] == 1]
            diagnosis_strings_chexpert = []
            for label in diagnosis_strings:
                assert label in mapping, f"Label {label} not found in mapping"
                if mapping[label] is not None:
                    assert mapping[label] in chexpert_labels, f"Label {mapping[label]} not found in chexpert labels"
                    diagnosis_strings_chexpert.append(mapping[label])

            diagnosis_strings_chexpert = list(set(diagnosis_strings_chexpert))

            if len(diagnosis_strings_chexpert) == 0:
                continue

            diagnosis_answer = ', '.join(diagnosis_strings_chexpert)
            diagnosis_annotation = {
                'images': [relative_path],
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
        return VINDR_CXR.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=VINDR_CXR.INPUT_SIZE, patch_size=VINDR_CXR.PATCH_SIZE, in_channels=VINDR_CXR.IN_CHANNELS),
        ]
