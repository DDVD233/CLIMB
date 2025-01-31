import json
import os
import ast
from typing import Any, List, Optional, Set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from tqdm import tqdm


class QUILT1M(Dataset):
    '''A dataset class for the QUILT-1M histopathology dataset'''

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'pathology', 'quilt_1m')
        self.image_dir = os.path.join(self.root, 'images')
        super().__init__()
        self.split: str = 'train' if train else 'valid'
        self.train = train
        self.file_list: Optional[pd.DataFrame] = None
        self.file_names: List[str] = []
        self.label_binarizer: Optional[MultiLabelBinarizer] = None
        self.unique_labels: Set[str] = set()
        self.build_index()

    def _parse_pathology_labels(self, label_str: str) -> List[str]:
        """Convert string representation of list to actual list"""
        if not isinstance(label_str, str) or label_str == 'nan':
            return []
        try:
            # Convert string to list and strip whitespace from each label
            labels = ast.literal_eval(label_str)
            clean_labels = [label.strip() for label in labels]
            # Remove 'Unknown' label
            return [label for label in clean_labels if label != 'Unknown']
        except:
            print(f"Warning: Failed to parse labels: {label_str}")
            return []

    def build_index(self):
        print('Building index...')

        # Read annotation file
        annotation_path = os.path.join(self.root, 'quilt_1M_lookup.csv')
        if not os.path.exists(annotation_path):
            raise RuntimeError(
                f"Annotation file not found at {annotation_path}"
            )

        # Load annotations
        annotations = pd.read_csv(annotation_path)

        # Convert pathology string to list and get unique labels
        annotations['pathology_list'] = annotations['pathology'].apply(self._parse_pathology_labels)
        print(f'Found {len(annotations)} annotations')

        # Filter out rows with empty labels
        annotations = annotations[annotations['pathology_list'].apply(len) > 0]
        print(f'Found {len(annotations)} annotations with labels')

        all_labels = [label for labels in annotations['pathology_list'] for label in labels]
        self.unique_labels = sorted(set(all_labels))

        # Initialize MultiLabelBinarizer with all possible labels
        self.label_binarizer = MultiLabelBinarizer(classes=self.unique_labels)
        self.label_binarizer.fit([self.unique_labels])

        # Convert labels to binary format
        label_matrix = self.label_binarizer.transform(annotations['pathology_list'])
        label_columns = [f'label_{label}' for label in self.unique_labels]
        label_df = pd.DataFrame(label_matrix, columns=label_columns)

        # Combine with original annotations
        self.file_list = pd.concat([annotations, label_df], axis=1)

        # Create 95:5 split
        # Get unique image paths to ensure we split by unique images
        unique_images = self.file_list.drop_duplicates('image_path')
        train_images, test_images = train_test_split(
            unique_images['image_path'],
            test_size=0.05,
            random_state=42
        )

        # Create split column
        self.file_list['split'] = 'train'
        self.file_list.loc[self.file_list['image_path'].isin(test_images), 'split'] = 'valid'

        # Filter based on split
        self.file_list = self.file_list[self.file_list['split'] == self.split].reset_index(drop=True)

        # Drop columns where image_path is not string
        self.file_list = self.file_list[self.file_list['image_path'].apply(lambda x: isinstance(x, str))]
        self.file_names = self.file_list['image_path'].tolist()

        # Print dataset statistics
        print(f'\nFound {len(self.file_names)} images for {self.split}')
        print('\nLabel distribution:')
        for label in self.unique_labels:
            count = self.file_list[f'label_{label}'].sum()
            print(f'{label}: {count} images')

    def __getitem__(self, index: int) -> Any:
        row = self.file_list.iloc[index]
        image_path = os.path.join(self.image_dir, row['image_path'])

        # Get binary labels
        label_columns = [f'label_{label}' for label in self.unique_labels]
        labels = row[label_columns].values

        return {
            'file_path': image_path,
            'labels': labels,
            'label_names': row['pathology_list'],
            'caption': row['caption']
        }

    def to_qa(self):
        """
        Convert the dataset to question answering format.
        Creates a multilabel classification question for each histopathology image.
        Includes list of all possible specialties in the question.
        """
        qa_data = []
        question_prefix = '<image>\nAbove is a histopathology image. '

        # Format all possible labels as a bullet point list
        label_choices = '\n'.join([label for label in sorted(self.unique_labels)])

        # Define the classification question with choices
        classification_question = ('What kind of histology image is this? '
                                   'Answer with one or multiple specialties from the following list, separated by commas:\n' +
                                   label_choices)

        for i in tqdm(range(len(self))):
            item = self[i]
            file_path = item['file_path']
            labels = sorted(item['label_names'])

            # Skip if no labels
            if not labels:
                continue

            # Create relative path for the image
            relative_path = os.path.relpath(file_path, self.root)

            # Join labels with commas
            label_text = ', '.join(labels)

            # Create QA item
            qa_item = {
                'images': [relative_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + classification_question},
                    {'from': 'gpt', 'value': label_text}
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