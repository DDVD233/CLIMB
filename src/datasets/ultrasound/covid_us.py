import json
import os
from typing import Any, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import subprocess
import gdown
import zipfile


class COVIDUS(Dataset):
    '''A dataset class for the COVID-US dataset'''

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'ultrasound', 'covid_us')
        super().__init__()
        if download:
            self.download()
            
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
        video_path = os.path.join(self.root, 'videos')

        # Check if required directory exists
        if not os.path.exists(video_path):
            raise RuntimeError(
                """
                Data not found in the expected location
                """
            )
        return video_path

    def _parse_filename(self, filename: str) -> tuple:
        """Helper method to parse video filename into components"""
        # Remove .mp4 extension
        name_without_ext = filename.rsplit('.', 1)[0]

        # Split by underscore
        parts = name_without_ext.split('_')

        # Get components
        id_num = parts[0]
        source = parts[1]
        label = parts[2]

        return id_num, source, label

    def _create_stratified_split(self, video_data: pd.DataFrame):
        """Helper method to create source-stratified train/test split"""
        # Drop videos with 'other' label
        video_data = video_data[video_data['label'] != 'other']

        # Create stratified split based on source
        train_df, test_df = train_test_split(
            video_data,
            test_size=0.25,
            random_state=42,
            stratify=video_data['source']
        )

        self.train_files = train_df
        self.test_files = test_df

    def build_index(self):
        print('Building index...')

        # Get all video files and their information
        video_files = []
        for file_name in os.listdir(self.index_location):
            if file_name.lower().endswith('.mp4'):
                file_path = os.path.join(self.index_location, file_name)
                id_num, source, label = self._parse_filename(file_name)
                video_files.append({
                    'file_path': file_path,
                    'id': id_num,
                    'source': source,
                    'label': label
                })

        # Create DataFrame
        video_df = pd.DataFrame(video_files)

        # Create split if not already done
        if self.train_files is None or self.test_files is None:
            self._create_stratified_split(video_df)

        # Select appropriate split
        self.file_list = self.train_files if self.train else self.test_files
        self.file_names = self.file_list['file_path'].tolist()

        # Print dataset statistics
        print(f'\nFound {len(self.file_names)} videos for {self.split}')
        print('\nLabel distribution:')
        print(self.file_list['label'].value_counts())
        print('\nSource distribution:')
        print(self.file_list['source'].value_counts())

    def __getitem__(self, index: int) -> Any:
        row = self.file_list.iloc[index]
        return {
            'file_path': row['file_path'],
            'label': row['label'],
            'source': row['source'],
            'id': row['id']
        }

    def to_qa(self):
        """
        Convert the dataset to question answering format.
        Creates a diagnosis classification question for each video.
        """
        qa_data = []
        question_prefix = '<video>\nAbove is a lung ultrasound video. '

        # Define the diagnosis question
        diagnosis_question = ('What is the diagnosis based on this lung ultrasound? '
                              'Answer with one word from the following options:\n'
                              'covid\npneumonia\nnormal')

        for i in range(len(self)):
            item = self[i]
            file_path = item['file_path']
            label = item['label']

            # Skip videos labeled as 'other'
            if label == 'other':
                continue

            # Create relative path for the video
            relative_path = os.path.relpath(file_path, self.root)

            # Create QA item
            qa_item = {
                'videos': [relative_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + diagnosis_question},
                    {'from': 'gpt', 'value': label}
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

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        file_id = "16HCezNef_bIyyHOgHqovkOYytpd4uPBM"
        gdown.download(f"https://drive.google.com/uc?id={file_id}",
                       os.path.join(self.root, "videos.zip"), quiet=False)
        with zipfile.ZipFile(os.path.join(self.root, "videos.zip"), "r") as zip_ref:
            zip_ref.extractall(self.root)
        annotation_ids = [("1s5tNVfcelKzF1NAJsnsJFMGlHMPxs9Qx", "annotation_train.jsonl"),
                          ("1oURVCLwHPRcFhyo8r7HXljwMcxXS_rXY", "annotation_valid.jsonl"),
                          ("1iXZspjTZr6PySMYWh-4gVn_8l-PpXZMu", "llava_med_test_results_finetuned1.jsonl"),
                          ("1NwHYQ-LfSV-v6PCvRLoUQOheLxAeQ4zB", "llava_med_test_results.jsonl")
                         ]
        for a_id, a_name in annotation_ids:
            gdown.download(f"https://drive.google.com/uc?id={a_id}",
                           os.path.join(self.root, a_name), quiet=False)
            
        #repo_url = "https://github.com/nrc-cnrc/COVID-US.git"
        #subprocess.run(["git", "clone", repo_url, self.root])
        print("Successfully downloaded dataset")
        
        

if __name__ == "__main__":
    d = COVIDUS(download=True, base_root='data')
    
