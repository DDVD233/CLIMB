import json
import os
from typing import Any, List, Optional

import pandas as pd
from torch.utils.data import Dataset


class BUSI(Dataset):
    '''A dataset class for the Breast Ultrasound Images dataset
    (https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
    '''

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'ultrasound', 'busi')
        else:
            self.root = base_root
        super().__init__()
        self.index_location = self.find_data()
        self.split: str = 'train' if train else 'valid'
        self.train = train
        self.file_list: Optional[pd.DataFrame] = None
        self.file_names: List[str] = []
        self.label: Optional[pd.DataFrame] = None
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        annotation_path = os.path.join(self.root, 'normal')
        # if no data is present, prompt the user to download it
        if not os.path.exists(annotation_path):
            raise RuntimeError(
                """
                'Visit https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset to download the data'
                """
            )
        return annotation_path

    def build_index(self):
        print('Building index...')

        # Initialize lists to store file paths and labels
        all_files = []
        labels = []

        # Define the class mapping
        class_mapping = {
            'normal': 0,
            'benign': 1,
            'malignant': 2
        }

        # Iterate through each class folder
        for class_name in ['normal', 'benign', 'malignant']:
            class_path = os.path.join(self.root, class_name)
            if not os.path.exists(class_path):
                continue

            # Find all PNG files in the folder that don't contain "mask"
            class_files = []
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith('.png') and 'mask' not in file_name.lower():
                    file_path = os.path.join(class_path, file_name)
                    class_files.append((file_path, class_mapping[class_name]))

            # Sort files alphabetically within each class
            class_files.sort(key=lambda x: os.path.basename(x[0]))

            # Split files into train (75%) and test (25%)
            split_idx = int(len(class_files) * 0.75)

            # Select either train or test files based on self.train
            selected_files = class_files[:split_idx] if self.train else class_files[split_idx:]

            # Add selected files to the main lists
            for file_path, label in selected_files:
                all_files.append(file_path)
                labels.append(label)

        # Create a DataFrame with file paths and labels
        self.file_list = pd.DataFrame({
            'file_path': all_files,
            'label': labels
        })

        # Store file names separately if needed
        self.file_names = self.file_list['file_path'].tolist()

        # Convert labels to a DataFrame
        self.label = pd.DataFrame(labels, columns=['label'])

        print(f'Found {len(self.file_names)} images for {self.split}:')
        for class_name, class_id in class_mapping.items():
            count = (self.label['label'] == class_id).sum()
            print(f'  {class_name}: {count} images')

    def __getitem__(self, index: int) -> Any:
        file_path = self.file_names[index]
        label = self.label.iloc[index]['label']

        return {
            'file_path': file_path,
            'label': label
        }

    def to_qa(self):
        """
        Convert the dataset to a question answering format.
        Each sample will contain a breast ultrasound image and a diagnostic question with the corresponding answer.
        """

        qa_data = []
        question_prefix = '<image>\nAbove is a breast ultrasound image. '

        # Define labels and create mapping
        labels = ['Normal', 'Benign', 'Malignant']
        idx_to_label = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}

        # Define the diagnosis question
        diagnosis_question = 'What is the diagnosis based on this breast ultrasound image? Answer with one word from the following:\n' + \
                             '\n'.join(labels)

        for i in range(len(self)):
            file_path = self.file_names[i]
            label_idx = self.label.iloc[i]['label']
            diagnosis = idx_to_label[label_idx]

            # Create relative path for the image
            relative_path = os.path.relpath(file_path, self.root)

            qa_item = {
                'images': [relative_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': diagnosis}
                ]
            }

            qa_data.append(qa_item)

        # Save the QA data to a JSONL file
        output_path = os.path.join(self.root, f'annotation_{self.split}.jsonl')
        with open(output_path, 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    def __len__(self) -> int:
        return self.label.shape[0]