import json
import os
from typing import Any, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class LC25000(Dataset):
    '''A dataset class for the LC25000 (Lung and Colon Histopathology) dataset'''

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'pathology', 'lc25000')
        super().__init__()
        # Define class names and their descriptions
        self.classes = {
            'colon_aca': 'colon adenocarcinoma',
            'colon_n': 'normal colon tissue',
            'lung_aca': 'lung adenocarcinoma',
            'lung_n': 'normal lung tissue',
            'lung_scc': 'lung squamous cell carcinoma'
        }
        self.index_location = self.find_data()
        self.split: str = 'train' if train else 'valid'
        self.train = train
        self.file_list: Optional[pd.DataFrame] = None
        self.file_names: List[str] = []
        self.train_files: Optional[pd.DataFrame] = None
        self.test_files: Optional[pd.DataFrame] = None
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        # Check if all required class folders exist
        for class_name in self.classes.keys():
            class_path = os.path.join(self.root, class_name)
            if not os.path.exists(class_path):
                raise RuntimeError(
                    f"""
                    Data folder {class_name} not found in {self.root}
                    """
                )
        return self.root

    def _create_split(self, image_data: pd.DataFrame):
        """Helper method to create train/test split"""
        # Create stratified split based on class
        train_df, test_df = train_test_split(
            image_data,
            test_size=0.25,
            random_state=42,
            stratify=image_data['label']
        )

        self.train_files = train_df
        self.test_files = test_df

    def build_index(self):
        print('Building index...')

        # Get all image files and their information
        image_files = []

        # Iterate through each class folder
        for class_name in self.classes.keys():
            class_path = os.path.join(self.root, class_name)

            # Get all images in the class folder
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(class_path, file_name)
                    image_files.append({
                        'file_path': file_path,
                        'label': class_name
                    })

        # Create DataFrame
        image_df = pd.DataFrame(image_files)

        # Create split if not already done
        if self.train_files is None or self.test_files is None:
            self._create_split(image_df)

        # Select appropriate split
        self.file_list = self.train_files if self.train else self.test_files
        self.file_names = self.file_list['file_path'].tolist()

        # Print dataset statistics
        print(f'\nFound {len(self.file_names)} images for {self.split}')
        print('\nClass distribution:')
        class_counts = self.file_list['label'].value_counts()
        for class_name in self.classes.keys():
            count = class_counts.get(class_name, 0)
            print(f'{class_name}: {count} images')

    def __getitem__(self, index: int) -> Any:
        row = self.file_list.iloc[index]
        return {
            'file_path': row['file_path'],
            'label': row['label']
        }

    def to_qa(self):
        """
        Convert the dataset to question answering format.
        Creates a 5-way classification question for each histopathology image.
        """
        qa_data = []
        question_prefix = '<image>\nAbove is a histopathology image. '

        # Define the diagnosis question and format the options
        diagnosis_options = '\n'.join([
            f'- {desc}' for desc in self.classes.values()
        ])
        diagnosis_question = ('What is the diagnosis based on this histopathology image? '
                              'Choose from the following options:\n' + diagnosis_options)

        for i in range(len(self)):
            item = self[i]
            file_path = item['file_path']
            label = item['label']

            # Create relative path for the image
            relative_path = os.path.relpath(file_path, self.root)

            # Create QA item
            qa_item = {
                'images': [relative_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': self.classes[label]}
                ]
            }
            qa_data.append(qa_item)

        # Save the QA data
        output_path = os.path.join(self.root, f'annotation_{self.split}.jsonl')
        with open(output_path, 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')

        return qa_data

    def __len__(self) -> int:
        return len(self.file_names)