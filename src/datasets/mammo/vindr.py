import json
import os
import shutil

import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.specs import Input2dSpec
from src.utils import count_files, get_pixel_array, LABEL_FRACS
from src.datasets.physionet import PhysioNetDownloader


class VINDR(Dataset):
    ''' A dataset class for the VinDR Mammography dataset. (https://www.physionet.org/content/vindr-mammo/1.0.0/)
    This dataset consists of left/right breast images from one of two views. 
    Each breast image is categorized on the BIRAD 1-5 scale, which communicates findings on presence/severity of lesions.
    Note: 
        1) You must register and manually download the data to this directory to use this dataset. Rename the folder you download "vindr". 
        Directions are available at the bottom of the above link after you make an account and click to complete the data use agreement.
    '''
    # Dataset information.
    NUM_CLASSES = 5
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1
    RANDOM_SEED = 0

    def __init__(self, base_root: str, download: bool = False, train: bool = True, finetune_size: str = None, file_list = None) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'mammo', 'vindr')
        if download:
            if file_list is not None:
                self.file_set = set(file_list)
            else:
                self.file_set = None
            self.download()
        
        self.split = 'train' if train else 'test'
        self.finetune_size = 0 if finetune_size is None else LABEL_FRACS[finetune_size]
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

    # save all dicom files as jpgs ahead of training for faster processing
    def dicom_to_jpg(self, df):
        fnames = df.iloc[:, [0, 1, 2]].to_numpy(dtype=str)
        for i in tqdm.tqdm(range(len(fnames))):
            # ignore race condition errors
            try:
                if not os.path.isdir(os.path.join(self.root, 'jpegs', fnames[i][0])):
                    os.makedirs(os.path.join(self.root, 'jpegs', fnames[i][0]))
            except OSError as e:
                if e.errno != 17:
                    print("Error:", e)
            dicom_path = os.path.join(self.root, 'images', fnames[i][0], fnames[i][2] + '.dicom')
            img_array = get_pixel_array(dicom_path)
            img = Image.fromarray(img_array)
            img.save(os.path.join(self.root, 'jpegs', fnames[i][0], fnames[i][2] + '.jpg'))

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.root, 'breast-level_annotations.csv')

        # get columns for study_id, series_id, image_id, breast_birads, and split
        df = pd.read_csv(index_file, header=0, usecols=[0, 1, 2, 7, 9])

        print('Converting DICOM to JPG')
        # Convert DICOM files to JPGs
        if os.path.isdir(os.path.join(self.root, 'jpegs')):
            if count_files(os.path.join(self.root, 'jpegs')) != 20000:
                shutil.rmtree(os.path.join(self.root, 'jpegs'))
                self.dicom_to_jpg(df)
        else:
            self.dicom_to_jpg(df)

        df_split_name = 'training' if self.split == 'train' else 'test'
        df = df.loc[df['split'] == df_split_name]  # use correct split

        # select subset of training data if finetuning
        if self.split == 'train' and self.finetune_size > 0:
            # get counts for every label
            unique_counts = df.iloc[:, 3].value_counts()
            train_df = pd.DataFrame(columns=df.columns)

            for label, count in unique_counts.items():
                # get 'finetune_size' labels for each class
                # if insufficient examples, use all examples from that class
                num_sample = min(self.finetune_size, count)
                train_rows = df.loc[df.iloc[:, 3] == label].sample(num_sample, random_state=VINDR.RANDOM_SEED)
                train_df = train_df.append(train_rows)

            df = train_df

        self.fnames = df.iloc[:, [0, 1, 2]].to_numpy(dtype=str)
        self.labels = df.iloc[:, 3].to_numpy(dtype=str)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'jpegs', self.fnames[index][0], self.fnames[index][2] + '.jpg')
        # Convert BIRAD 1-5 classification to class label (0-4)
        label = int(self.labels[index][-1]) - 1
        img = Image.open(img_path)
        img = self.transforms(img)

        # Add image-dependent padding
        dim_gap = img.shape[1] - img.shape[2]
        pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
        img = transforms.Pad((pad1, 0, pad2, 0))(img)

        return index, img, label

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """
        qa_data = []
        question_prefix = '<image>\nAbove is a mammography X-ray image of a patient. '
        for index in range(len(self)):
            # Convert BIRAD 1-5 classification to class label (0-4)
            label = int(self.labels[index][-1])

            diagnosis_question = ('What is the BIRAD classification of the patient in the X-ray image? Answer with one '
                                  'number. Answer with one phrase from the following: BIRAD 1\nBIRAD 2\nBIRAD 3\nBIRAD 4\nBIRAD 5\n')
            assert label in range(1, 6), f"Invalid label: {label}"
            diagnosis_answer = f"BIRAD {label}"
            relative_path = os.path.join('jpegs', self.fnames[index][0], self.fnames[index][2] + '.jpg')

            qa_data.append({
                'images': [relative_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': diagnosis_answer}
                ]
            })

        with open(os.path.join(self.root, f'annotation_{self.split}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

    def __len__(self):
        return self.fnames.shape[0]

    @staticmethod
    def num_classes():
        return VINDR.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=VINDR.INPUT_SIZE, patch_size=VINDR.PATCH_SIZE, in_channels=VINDR.IN_CHANNELS),
        ]

    def download(self):
        downloader = PhysioNetDownloader("https://www.physionet.org/files/vindr-mammo/1.0.0/")

        required_files = [
            "LICENSE.txt",
            "SHA256SUMS.txt",
            "breast-level_annotations.csv",
            "finding_annotations.csv",
            "metadata.csv"
        ]

        for file in required_files:
            remote_path = file
            local_path = os.path.join(self.root, file)
            
            if not os.path.exists(local_path):
                print(f"Downloading {file}...")
                success = downloader.download_file(remote_path, local_path)
                if not success:
                    raise RuntimeError(f"Failed to download {file}")

        metadata = pd.read_csv(os.path.join(self.root, required_files[3]))

        required_files = []

        for i in range(len(metadata['study_id'])):
            study_id = str(metadata['study_id'][i])
            image_id = str(metadata['image_id'][i])
            if self.file_set is None or study_id in self.file_set or image_id in self.file_set or 'images' in self.file_set:
                required_files.append(os.path.join('images', study_id, image_id) + '.dicom')

        for file in required_files:
            remote_path = file
            local_path = os.path.join(self.root, file)
            
            if not os.path.exists(local_path):
                print(f"Downloading {file}...")
                success = downloader.download_file(remote_path, local_path)
                if not success:
                    raise RuntimeError(f"Failed to download {file}")

        print("Download completed successfully!")



if __name__ == "__main__":
    d = VINDR(download=True, base_root='data', file_list=['0025a5dc99fd5c742026f0b2b030d3e9',
                                                '16e58fc1d65fa7587247e6224ee96527',
                                                '7fc1f1bb8bb1a7efaf7104e49c4d8b86'])

