import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

EVENT_LABELS = {
    'Gene_expression': 0,
    'Positive_regulation': 1,
    'Negative_regulation': 2,
    'Speculation': 3,
    'Binding': 4,
    'Localization': 5,
    'Transcription': 6,
    'Negation': 7,
    'Regulation': 8,
    'Entity': 9,
    'Phosphorylation': 10,
    # more will be added
}


class GENIA(Dataset):
    '''A dataset class for the GENIA dataset (https://www.kaggle.com/datasets/nishanthsalian/genia-biomedical-event-dataset).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'text', 'GENIA')
        else:
            self.root = base_root
        super().__init__()
        self.split = 'train' if train else 'valid'
        self.index_location = self.find_data()
        self.texts: Optional[np.ndarray] = None
        self.label: Optional[np.ndarray] = None
        self.trigger_locs: Optional[np.ndarray] = None
        self.trigger_words: Optional[np.ndarray] = None
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        annotation_path = os.path.join(
            self.root, 'train_data.csv' if self.split == 'train' else 'test_data.csv')
        # if no data is present, prompt the user to download it
        if not os.path.exists(annotation_path):
            raise RuntimeError(
                """
                'Visit https://www.kaggle.com/datasets/nishanthsalian/genia-biomedical-event-dataset to download the data'
                """
            )
        return annotation_path

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.index_location)
        df = pd.read_csv(index_file)

        self.texts = np.array(df['Sentence'].tolist())  # (8666,)
        event_types = [
            self.get_event_indices(events) if pd.notna(events) else []
            for events in df['EventType'].tolist()
        ]
        self.label = np.array(event_types, dtype=object) # type set to object since one sentence can have multiple events
        self.trigger_locs = np.array(
            [split_and_filter(loc) if pd.notna(loc) else []
             for loc in df['TriggerWordLoc'].tolist()],
            dtype=object)
        self.trigger_words = np.array(
            [split_and_filter(tw) if pd.notna(tw) else []
             for tw in df['TriggerWord'].tolist()],
            dtype=object)

    def __len__(self) -> int:
        return self.label.shape[0]
    
    def get_event_indices(self, events):
        events = split_and_filter(events)
        indices = []
        for e in events:
            if e not in EVENT_LABELS:
                EVENT_LABELS[e] = self.num_classes()
            indices.append(EVENT_LABELS[e])
        return indices

    def __getitem__(self, index: int) -> Any:
        data = {
            'sentence': self.texts[index],
            'labels': self.label[index],
            'trigger_words': self.trigger_words[index],
            'trigger_word_locs': self.trigger_locs[index]
        }

        return data

    @staticmethod
    def num_classes():
        return len(EVENT_LABELS)
    
def split_and_filter(data):
    return [elm for elm in data.split(';') if elm]

genia = GENIA('/Users/malindalu/Desktop/high_modality/')
print(genia[0])
print(genia.num_classes())
print(EVENT_LABELS)
