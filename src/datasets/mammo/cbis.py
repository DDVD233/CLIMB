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
from src.utils import count_files, get_pixel_array


class CBIS(Dataset):
    ''' A dataset class for CBIS-DDSM: Breast Cancer Image Dataset.
    (https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
    (https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
    This dataset consists of single breast images, either left or right breast, from one of two views (CC or MLO). 
    Each breast will be categorized on the BIRAD 1-5 scale, which communicates findings on presence/severity of lesions.
    Note: 
        1) Additional preprocessing was used to convert lesion-level BIRAD assessments into breast-level assessments.
        2) You must manually download the data zip from the Kaggle link above into this directory. Rename the the folder you extract from
        the zip file as "cbis". It should contain folders "csv" and "jpeg". 
    '''
    # Dataset information.
    NUM_CLASSES = 5
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'mammo', 'cbis')
        self.split = 'train' if train else 'test'  # use dataset's test split for validation

        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.2003], [1.0]),
            ]
        )
        self.build_index()

    # save all dicom files as jpgs ahead of training for faster processing
    def dicom_to_jpg(self):
        train_mass_info = os.path.join(self.root, 'csv', 'mass_case_description_train_set.csv')
        train_calc_info = os.path.join(self.root, 'csv', 'calc_case_description_train_set.csv')
        val_mass_info = os.path.join(self.root, 'csv', 'mass_case_description_test_set.csv')
        val_calc_info = os.path.join(self.root, 'csv', 'calc_case_description_test_set.csv')
        metadata = os.path.join(self.root, 'metadata.csv')
        # Get columns for patient_id, left or right breast, image view, assessment, and dicom file path
        df = pd.read_csv(train_mass_info, header=0, usecols=[0, 2, 3, 8, 11])
        # df = df.append(pd.read_csv(train_calc_info, header=0, usecols=[0, 2, 3, 8, 11]))
        # df = df.append(pd.read_csv(val_mass_info, header=0, usecols=[0, 2, 3, 8, 11]))
        # df = df.append(pd.read_csv(val_calc_info, header=0, usecols=[0, 2, 3, 8, 11]))
        df = pd.concat([df, pd.read_csv(train_calc_info, header=0, usecols=[0, 2, 3, 8, 11])])
        df = pd.concat([df, pd.read_csv(val_mass_info, header=0, usecols=[0, 2, 3, 8, 11])])
        df = pd.concat([df, pd.read_csv(val_calc_info, header=0, usecols=[0, 2, 3, 8, 11])])
        metadata_df = pd.read_csv(metadata)

        path_mapping = {}
        for row in metadata_df.iterrows():
            # annotation_path is "Subject ID/Study UID/Series UID"
            data = row[1]
            annotation_path = f"{data['Subject ID']}/{data['Study UID']}/{data['Series UID']}"
            real_path = data['File Location'][2:]  # remove leading "./"
            path_mapping[annotation_path] = real_path

        for i in tqdm.tqdm(range(len(df))):
            # get file path to dicom, excluding file name
            file_path = df.iloc[i][4].rsplit('/', 1)[0]
            real_path = path_mapping[file_path]
            # ignore race condition errors
            try:
                if not os.path.isdir(os.path.join(self.root, 'jpegs', file_path)):
                    os.makedirs(os.path.join(self.root, 'jpegs', file_path))
            except OSError as e:
                if e.errno != 17:
                    print("Error:", e)
            dicom_path = os.path.join(self.root, real_path, '1-1.dcm')
            img_array = get_pixel_array(dicom_path)
            img = Image.fromarray(img_array)
            img.save(os.path.join(self.root, 'jpegs', file_path, '1-1.jpg'))

    def build_index(self):
        print('Building index...')
        # Convert DICOM files to JPGs
        if os.path.isdir(os.path.join(self.root, 'jpegs')):
            if count_files(os.path.join(self.root, 'jpegs')) != 3103:
                # shutil.rmtree(os.path.join(self.root, 'jpegs'))
                # self.dicom_to_jpg()
                print('WARNING: File count mismatch.')
        else:
            self.dicom_to_jpg()

        mass_info = os.path.join(self.root, 'csv', f'mass_case_description_{self.split}_set.csv')
        calc_info = os.path.join(self.root, 'csv', f'calc_case_description_{self.split}_set.csv')
        df = pd.read_csv(mass_info, header=0, usecols=[0, 2, 3, 8, 11])
        # df = df.append(pd.read_csv(calc_info, header=0, usecols=[0, 2, 3, 8, 11]))
        df = pd.concat([df, pd.read_csv(calc_info, header=0, usecols=[0, 2, 3, 8, 11])])
        # Combine patient id, L/R and CC/MLO into image name, which we will filter for duplicates
        df['ImageName'] = df["patient_id"] + df["left or right breast"] + df['image view']

        # Only use birad 1-5, since birad 0 indicates an inconclusive result/further imaging needed
        df = df.loc[df['assessment'] != 0]
        # Drop duplicate images from csv, keeping highest birad classification noted for each image
        df['assessment'] = pd.to_numeric(df['assessment'])
        df = df.sort_values('assessment', ascending=False).drop_duplicates('ImageName').sort_index()
        self.fnames = df['image file path'].to_numpy(dtype=str)
        self.labels = df['assessment'].to_numpy(dtype=str)
        print('Done \n\n')

    def to_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """

        qa_data = []

        question_prefix = '<image>\nAbove is a mammography X-ray image of a patient. '

        for i in range(len(self)):
            label: np.ndarray = self.labels[i]
            fname = os.path.join('jpegs', self.fnames[i].rsplit('/', 1)[0], '1-1.jpg')

            # Create a question for each label
            diagnosis_question = ('What is the BIRAD classification of the patient in the X-ray image? Answer with one '
                                  'number. Answer with one phrase from the following: BIRAD 1\nBIRAD 2\nBIRAD 3\nBIRAD 4\nBIRAD 5\n')
            assert label in ['1', '2', '3', '4', '5'], f'Invalid label: {label}'
            diagnosis_answer = f"BIRAD {label}"

            qa_data.append({
                'images': [fname],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': diagnosis_answer}
                ]
            })

        with open(os.path.join(self.root, f'annotation_{self.split}.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'jpegs', self.fnames[index].rsplit('/', 1)[0], '1-1.jpg')
        # Convert BIRAD 1-5 classification to class label (0-4)
        label = int(self.labels[index][-1]) - 1
        img = Image.open(img_path)
        img = self.transforms(img)

        # Add image-dependent padding
        dim_gap = img.shape[1] - img.shape[2]
        pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
        img = transforms.Pad((pad1, 0, pad2, 0))(img)

        return index, img, label

    def __len__(self):
        return self.fnames.shape[0]

    @staticmethod
    def num_classes():
        return CBIS.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=CBIS.INPUT_SIZE, patch_size=CBIS.PATCH_SIZE, in_channels=CBIS.IN_CHANNELS),
        ]
