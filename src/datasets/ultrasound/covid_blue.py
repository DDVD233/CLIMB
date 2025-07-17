import json
import os
from typing import Any, List, Optional
import pandas as pd
from torch.utils.data import Dataset

import subprocess
import gdown
import zipfile


class COVIDBLUES(Dataset):
    '''A dataset class for the COVID Bluepoint Lung Ultrasound (BLUES) dataset'''

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'ultrasound', 'COVID-BLUES')
        super().__init__()
        if download:
            self.download()

        self.index_location = self.find_data()
        self.split: str = 'train' if train else 'valid'
        self.train = train
        self.file_list: Optional[pd.DataFrame] = None
        self.file_names: List[str] = []
        self.clinical_data: Optional[pd.DataFrame] = None
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        video_path = os.path.join(self.root, 'lus_videos')
        clinical_data_path = os.path.join(self.root, 'clinical_variables.csv')

        # Check if required files exist
        if not os.path.exists(video_path) or not os.path.exists(clinical_data_path):
            raise RuntimeError(
                """
                Visit https://github.com/NinaWie/COVID-BLUES to download the data
                """
            )
        return video_path

    def build_index(self):
        print('Building index...')

        # Read clinical data
        clinical_data_path = os.path.join(self.root, 'clinical_variables.csv')
        self.clinical_data = pd.read_csv(clinical_data_path)

        # Get all video files
        video_files = []
        for file_name in os.listdir(self.index_location):
            if file_name.lower().endswith('.mp4'):
                file_path = os.path.join(self.index_location, file_name)
                # Extract patient ID from filename (between first and second underscore)
                patient_id = int(file_name.split('_')[1])
                video_files.append((file_path, patient_id))

        # Sort files by patient ID
        video_files.sort(key=lambda x: x[1])

        # Split files into train (75%) and test (25%) by patient ID
        unique_patients = sorted(list(set(x[1] for x in video_files)))
        split_idx = int(len(unique_patients) * 0.75)
        train_patients = set(unique_patients[:split_idx])
        test_patients = set(unique_patients[split_idx:])

        # Select files based on split
        selected_files = []
        for file_path, patient_id in video_files:
            if (patient_id in train_patients and self.train) or \
                    (patient_id in test_patients and not self.train):
                selected_files.append((file_path, patient_id))

        # Create DataFrame with file paths and patient IDs
        self.file_list = pd.DataFrame(selected_files, columns=['file_path', 'patient_id'])
        self.file_names = self.file_list['file_path'].tolist()

        print(f'Found {len(self.file_names)} videos for {self.split}')
        print(f'Number of unique patients: {len(self.file_list["patient_id"].unique())}')

    def __getitem__(self, index: int) -> Any:
        file_path = self.file_names[index]
        patient_id = self.file_list.iloc[index]['patient_id']

        # Get clinical data for this patient
        patient_data = self.clinical_data[self.clinical_data['patient_id'] == patient_id].iloc[0]

        return {
            'file_path': file_path,
            'patient_id': patient_id,
            'covid_status': patient_data['cov_test'],
            'age': patient_data['pat_age']
        }

    def to_qa(self):
        """
        Convert the dataset to two question answering formats:
        1. COVID status question
        2. Patient age question
        """
        qa_covid_data = []
        qa_age_data = []

        question_prefix = '<video>\nAbove is a lung ultrasound video. '

        # Define the COVID question and possible answers
        covid_question = 'Based on the ultrasound, does the patient have COVID?'
        covid_answers = {0: 'No COVID', 1: 'Has COVID'}

        # Define the age question
        age_question = 'Based on the ultrasound, how old is the patient?'

        for i in range(len(self)):
            item = self[i]
            file_path = item['file_path']
            covid_status = item['covid_status']
            age = item['age']

            # Create relative path for the video
            relative_path = os.path.relpath(file_path, self.root)

            # Create COVID QA item
            covid_qa_item = {
                'videos': [relative_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + covid_question},
                    {'from': 'gpt', 'value': covid_answers[covid_status]}
                ]
            }
            qa_covid_data.append(covid_qa_item)

            # Create Age QA item
            age_qa_item = {
                'images': [relative_path],
                'explanation': '',
                'conversations': [
                    {'from': 'human', 'value': question_prefix + age_question},
                    {'from': 'gpt', 'value': str(int(age))}
                ]
            }
            qa_age_data.append(age_qa_item)

        # Save the COVID QA data
        covid_output_path = os.path.join(self.root, f'annotation_{self.split}.jsonl')
        with open(covid_output_path, 'w') as f:
            for qa in qa_covid_data:
                f.write(json.dumps(qa) + '\n')

        # Save the Age QA data
        age_output_path = os.path.join(self.root, f'annotation_age_{self.split}.jsonl')
        with open(age_output_path, 'w') as f:
            for qa in qa_age_data:
                f.write(json.dumps(qa) + '\n')

        return qa_covid_data, qa_age_data

    def __len__(self) -> int:
        return len(self.file_names)

    def download(self):
        repo_url = "https://github.com/NinaWie/COVID-BLUES.git"
        subprocess.run(["git", "clone", repo_url, self.root])
        annotation_ids = [("1-BphivsAXCY69EQC8EVEje66_qRCOHob", "annotation_train.jsonl"),
                          ("1LESKYHEPOpaR60WZzIVqJKp7PjtH8BCB", "annotation_valid.jsonl"),
                          ("1onN3o-04oHuP-c57jieduXN_0moeN8yW", "annotation_age_train.jsonl"),
                          ("19OIKqpdjOwfKIzpKvHh2ot16hKCy60Dy", "annotation_age_valid.jsonl")
                          ]
        for a_id, a_name in annotation_ids:
            gdown.download(f"https://drive.google.com/uc?id={a_id}",
                           os.path.join(self.root, a_name), quiet=False)
        print("Successfully downloaded dataset")


if __name__ == "__main__":
    d = COVIDBLUES(download=True, base_root='data')