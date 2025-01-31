import json
import os
import random
import re

import pandas as pd
from torch.utils.data import Dataset


class Cholec(Dataset):
    '''
    A dataset class for the Cholec dataset, which includes videos and corresponding phase and tool annotations.
    Each video has an associated timestamp.txt file indicating the phase at each timestamp, a phase annotation file,
    and a tool annotation file with tool presence information at each frame.
    '''

    RANDOM_SEED = 0
    TRAIN_SPLIT_FRAC = 0.8
    TOOLS = ['Grasper', 'Bipolar', 'Hook', 'Scissors',
             'Clipper', 'Irrigator', 'Specimen Bag']
    PHASES = {'Preparation':'Preparation', 
              'CalotTriangleDissection':'Calot Triangle Dissection',
              'ClippingCutting':'Clipping Cutting', 
              'GallbladderDissection':'Gallbladder Dissection', 
              'GallbladderPackaging':'Gallbladder Packaging', 
              'CleaningCoagulation':'Cleaning Coagulation', 
              'GallbladderRetraction':'Gallbladder Retraction'}

    def __init__(self, base_root: str, train: bool = True, finetune_size: str = None) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'endoscopic', 'cholec80')
        self.split = 'train' if train else 'valid'
        self.finetune_size = finetune_size
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.build_index()

    def build_index(self):
        print('Building index...')
        video_dir = os.path.join(self.root, 'videos')
        # Load phase and tool annotations
        # video_paths = [os.path.join(video_dir, f) for f in os.listdir(
        #     video_dir) if f.endswith('.mp4')]
        video_paths = [os.path.join(video_dir, re.search(r"(video\d+)", filename).group(0) + '.mp4') for filename in os.listdir(
            video_dir) if filename.endswith('.txt')]
        
        # video_paths = video_paths[:4] # for testing

        num_sample = int(Cholec.TRAIN_SPLIT_FRAC * len(video_paths))
        random.seed(Cholec.RANDOM_SEED)
        training_videos = random.sample(video_paths, num_sample)
        if self.split == 'train':
            video_paths = training_videos
        else:
            video_paths = [video for video in video_paths if video not in training_videos]

        self.phase_data = {video:{} for video in video_paths}
        self.tool_data = {video:{} for video in video_paths}
        self.video_names = []

        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.video_names.append(video_name)

            # Load timestamp, tool, and phase annotations
            phase_path = os.path.join(
                self.root, 'phase_annotations', f'{video_name}-phase.txt')
            tool_path = os.path.join(
                self.root, 'tool_annotations', f'{video_name}-tool.txt')
            timestamp_path = os.path.join(
                video_dir, f'{video_name}-timestamp.txt')

            phase_annotations = pd.read_csv(phase_path, delimiter='\t')
            tool_annotations = pd.read_csv(tool_path, sep='\t')
            timestamp_annotations = pd.read_csv(timestamp_path, delimiter='\t')

            for index, row in phase_annotations.iterrows():
                frame = row['Frame']
                phase = row['Phase']

                timestamp = timestamp_annotations.iloc[index]['Frame']
                # Extract tool annotations for this frame
                tool_info = tool_annotations.loc[tool_annotations['Frame'] == frame]
                # Check if any matching rows are found for this frame
                if not tool_info.empty:
                    tool_info = tool_info.iloc[0].to_dict()
                    tool_in_video = []
                    for tool in self.TOOLS:
                        if tool_info[tool if tool!= 'Specimen Bag' else 'SpecimenBag'] == 1:
                            tool_in_video.append(tool)
                    self.tool_data[video_path][timestamp] = [frame, tool_in_video]

                self.phase_data[video_path][timestamp] = [frame, [phase]]
            print(f'Built {video_name}')
        print('Done\n\n')

    def to_phase_qa(self):
        """
        Convert the dataset to a question answering dataset.
        """
        print('Creating phase QA...')
        qa_data = []
        question_prefix = '<video>\nAbove is a video of a cholecystectomy surgery. '

        for i in range(len(self)):
            file_path = os.path.join(self.root, 'videos', self.video_names[i] + '.mp4')
            relative_path = os.path.relpath(file_path, self.root)
            
            phase_per_interval = Cholec.sample_intervals(self.phase_data[file_path],90)

            for interval in phase_per_interval:
                diagnosis_question = ('Identify the phases of cholecystectomy surgery that are present in the video, choosing from '
                                      'Preparation, Calot Triangle Dissection, Clipping Cutting, Gallbladder Dissection, Gallbladder Packaging, Cleaning Coagulation, and Gallbladder Retraction.'
                                    )
                diagnosis_answer = ", ".join(str(self.PHASES[item]) for item in phase_per_interval[interval])
                diagnosis_annotation = {
                    'video': [relative_path],
                    'explanation': '',
                    'interval': [interval[0], interval[1]],
                    'conversations': [
                        {'from': 'human', 'value': question_prefix + diagnosis_question},
                        {'from': 'gpt', 'value': diagnosis_answer}
                    ]
                }
                qa_data.append(diagnosis_annotation)

        with open(os.path.join(self.root, f'annotation_{self.split}_phase.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')
        print("Done")
        
        return qa_data
    
    @staticmethod
    def sample_intervals(data, num_intervals):
        # Sort the dictionary by timestamps (keys)
        sorted_data = dict(sorted(data.items()))
        timestamps = list(sorted_data.keys())
        values = [val[1] for val in list(sorted_data.values())]
        
        # Define the interval size based on the number of intervals
        interval_size = len(timestamps) // num_intervals
        results = {}
        
        for i in range(num_intervals):
            start_idx = i * interval_size
            end_idx = (i + 1) * interval_size if i != num_intervals - 1 else len(timestamps)
            
            # Sample values within the interval
            interval_values = [item for sublist in values[start_idx:end_idx] for item in sublist]            
            # Store the result with the interval as the key (using start timestamp as representative)
            interval_key = (timestamps[start_idx], timestamps[end_idx-1])
            results[interval_key] = set(interval_values)

        return results
    
    def to_tool_qa(self):
        print('Creating tools QA...')
        qa_data = []
        question_prefix = '<video>\nAbove is a video of a cholecystectomy surgery. '

        for i in range(len(self)):
            file_path = os.path.join(self.root, 'videos', self.video_names[i] + '.mp4')
            relative_path = os.path.relpath(file_path, self.root)
            
            tools_per_interval = Cholec.sample_intervals(self.tool_data[file_path],90)

            for interval in tools_per_interval:
                diagnosis_question = ('Identify the tools present in the video, choosing from '
                                    'Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag, and None.')
                diagnosis_answer = ", ".join(str(item) for item in tools_per_interval[interval]) or 'None'
                diagnosis_annotation = {
                    'video': [relative_path],
                    'explanation': '',
                    'interval': [interval[0], interval[1]],
                    'conversations': [
                        {'from': 'human', 'value': question_prefix + diagnosis_question},
                        {'from': 'gpt', 'value': diagnosis_answer}
                    ]
                }
                qa_data.append(diagnosis_annotation)

        with open(os.path.join(self.root, f'annotation_{self.split}_tool.jsonl'), 'w') as f:
            for qa in qa_data:
                f.write(json.dumps(qa) + '\n')
        print("Done")
                
        return qa_data

    def __len__(self):
        return len(self.video_names)


if __name__ == '__main__':
    cholec = Cholec('/Users/malindalu/Desktop/high_modality/', train=True)
    cholec.to_phase_qa()
    cholec.to_tool_qa()
    cholec = Cholec('/Users/malindalu/Desktop/high_modality/', train=False)
    cholec.to_phase_qa()
    cholec.to_tool_qa()
