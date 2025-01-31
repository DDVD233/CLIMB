import json
import math
import os
import shutil
import cv2
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.specs import Input2dSpec
from src.utils import count_files, get_pixel_array


def apply_window_level(data, window=700, level=80):
    """
    Apply radiological window/level adjustment
    window: controls the range of values displayed (contrast)
    level: controls the center of the window (brightness)
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
    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    try:
        # Apply stronger CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Additional brightness adjustment
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=30)
        return enhanced
    except:
        return image


class BRIXIA(Dataset):
    '''A dataset class for the Brixia COVID-19 dataset.
    This dataset consists of chest X-rays with COVID-19 severity scores.
    Each image is categorized on a global Brixia score scale.
    '''
    # Dataset information
    NUM_CLASSES = 6  # Assuming BrixiaScoreGlobal ranges from 0-5
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'chest_xray', 'brixia')
        self.split = 'train' if train else 'test'
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.1180], [1]),
            ]
        )
        self.build_index()

    def dicom_to_jpg(self, df):
        fnames = df['Filename'].to_numpy(dtype=str)
        for fname in tqdm.tqdm(fnames):
            # ignore race condition errors
            try:
                if not os.path.isdir(os.path.join(self.root, 'jpegs')):
                    os.makedirs(os.path.join(self.root, 'jpegs'))
            except OSError as e:
                if e.errno != 17:
                    print("Error:", e)

            dicom_path = os.path.join(self.root, 'dicom_clean', fname)
            img_array = get_pixel_array(dicom_path)

            # Handle MONOCHROME1 (inverted) images
            # photometric = df.loc[df['Filename'] == fname, 'PhotometricInterpretation'].iloc[0]
            # if photometric == 'MONOCHROME1':
            #     img_array = np.invert(img_array)

            # Apply window/level adjustment with brighter settings
            img_array = apply_window_level(img_array, window=512,
                                           level=256)  # Wider window and higher level for better visibility

            # Apply enhancement
            img_array = denoise_and_enhance(img_array)

            img = Image.fromarray(img_array)
            jpg_name = fname.replace('.dcm', '.jpg')
            img.save(os.path.join(self.root, 'jpegs', jpg_name))

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.root, 'metadata_global_v2.csv')

        # Read CSV with semicolon delimiter
        df = pd.read_csv(index_file, sep=';')

        print('Converting DICOM to JPG')
        # Convert DICOM files to JPGs if not already done
        if os.path.isdir(os.path.join(self.root, 'jpegs')):
            if count_files(os.path.join(self.root, 'jpegs')) != len(df):
                shutil.rmtree(os.path.join(self.root, 'jpegs'))
                self.dicom_to_jpg(df)
        else:
            self.dicom_to_jpg(df)

        # Split based on ConsensusTestset
        is_test = df['ConsensusTestset'] == 1
        df = df[is_test if self.split == 'test' else ~is_test]

        self.metadata = df
        self.fnames = df['Filename'].to_numpy(dtype=str)
        self.labels = df['BrixiaScoreGlobal'].to_numpy(dtype=int)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'jpegs', self.fnames[index].replace('.dcm', '.jpg'))
        label = self.labels[index]
        img = Image.open(img_path)
        img = self.transforms(img)

        # Add image-dependent padding
        dim_gap = img.shape[1] - img.shape[2]
        pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
        img = transforms.Pad((pad1, 0, pad2, 0))(img)

        return index, img, label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset with metadata.
        """
        qa_data = []
        question_prefix = '<image>\nAbove is a chest X-ray image of a COVID-19 patient. '

        for index in range(len(self)):
            row = self.metadata.iloc[index]

            diagnosis_question = ('Based on this chest X-ray and the provided metadata, '
                                  'what is the score indicating COVID-19 severity on a scale of 1-3? '
                                  'Answer with just the score number.')

            label = self.labels[index]
            avg_score = math.ceil(row['BrixiaScoreGlobal'] / 6)
            diagnosis_answer = f"{avg_score}"

            relative_path = os.path.join('jpegs', self.fnames[index].replace('.dcm', '.jpg'))

            qa_data.append({
                'images': [relative_path],
                'sex': row['Sex'],
                'age_group': int(row['AgeAtStudyDateFiveYear']),
                'manufacturer': row['Manufacturer'],
                'study_date': int(row['StudyDate']),
                'score': int(row['BrixiaScoreGlobal']),  # 0-18
                'score_avg': avg_score,  # 1-3
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': diagnosis_answer}
                ]
            })

        # Save to JSONL file
        output_file = os.path.join(self.root, f'annotation_{self.split}.jsonl')
        with open(output_file, 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

    def __len__(self):
        return len(self.fnames)

    @staticmethod
    def num_classes():
        return BRIXIA.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=BRIXIA.INPUT_SIZE, patch_size=BRIXIA.PATCH_SIZE, in_channels=BRIXIA.IN_CHANNELS),
        ]