import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.specs import Input1dSpec

# using the training (70%) data
_MARGINAL_CLASS_DIST = {0: 1029836, 1: 133999, 2: 1455953, 3: 419273, 4: 525718}
import glob
import os
import xml.etree.ElementTree as ET

import mne
import numpy as np
import pandas as pd
import tqdm

DATASET = 'SHHS'
CHANNELS = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']  # https://arxiv.org/pdf/1910.06100.pdf
CLASSES = ['W', 'N1', 'N2', 'N3', 'R']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}  # R is called "5" in the oringinal data, but use 4 here
CLASS_FREQ = {"W": 221, "N1": 126, "N2": 281, "N3": 214, "R": 62}  # Basing on patient_8_1
EPOCH_LENGTH = 30 * 125  # 30 seconds * 125 Hz

_MIN = -0.000125
_MAX = 0.000125


class SHHS(Dataset):
    '''A dataset class for the SHHS dataset (The Sleep Heart Health Study),
     a multi-center cohort study implemented by the National Heart Lung & Blood Institute 
     to determine the cardiovascular and other consequences of sleep-disordered breathing.

    https://sleepdata.org/datasets/shhs
    
    Note that you must register an NSRR account and go to the Request Access page to gain access, then 
    download the study 1 data using NSRR gem to your base_root directory 
    then you must extract and rename to SHHS, then you can run preprocess_shhs.py 
    to preprocess the data from edfs to pkl and numpy files (30 second epochs)
    '''

    # Dataset information.

    NUM_CLASSES = 5  # 5 different clinical observations for classification
    SEGMENT_SIZE = 30
    IN_CHANNELS = 2  # 12-lead ECG readings
    RANDOM_SEED = 0
    TRAIN_SPLIT_FRAC = 0.8
    DATASET = DATASET
    CLASSES = CLASSES
    LABEL_MAP = LABEL_MAP
    EPOCH_LENGTH = EPOCH_LENGTH
    MEASUREMENTS_PER_EXAMPLE = 30 * 125
    CLASS_FREQ = _MARGINAL_CLASS_DIST

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:

        super().__init__()
        self.root = os.path.join(base_root, 'eeg', 'SHHS/', 'processed')
        self.DATA_PATH = os.path.join(base_root, 'eeg', 'SHHS/')
        split_ratios = [self.TRAIN_SPLIT_FRAC * 100, (1 - self.TRAIN_SPLIT_FRAC) * 100]
        patients = [int(x) for x in os.listdir(self.root)]
        self.split = 'train' if train else 'val'
        self.pids = self.get_split_indices(split_ratios, len(patients), names=['train', 'val'])[self.split]
        self.pids = sorted([patients[i] for i in self.pids])
        #  if filter_pids: self.pids = [pid for pid in self.pids if pid in set(filter_pids)]

        self.labels = []
        self.groups = []
        self._pid_to_start_cidx = {}
        idx = 0
        for pid in self.pids:
            _curr_label = load_one_pid_label(self.DATA_PATH, pid)
            self._pid_to_start_cidx[pid] = idx
            for _cidx, y in enumerate(_curr_label):
                self.labels.append(y)
                idx += 1
            self.groups.extend([pid] * len(_curr_label))
        self.labels = np.asarray(self.labels)
        self.groups = np.asarray(self.groups)

        #normalizing constants
        self.percs = {}
        self.norm = None
        for pid in self.pids:
            if self.norm is None:
                self.percs[pid] = (_MIN, _MAX)
            else:
                _perc = load_one_pid_perc(self.DATA_PATH, pid)['both']
                assert isinstance(self.norm, int)
                self.percs[pid] = (_perc[self.norm], _perc[100 - self.norm])

        self.n = len(self.labels)
        self.alternative_X = None

    def __len__(self):
        return self.n

    def idx2pid(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.n:
            raise IndexError("%d is out of range (%d elements)" % (idx, self.n))
        return self.groups[idx]

    def read_processed_data(self, pid, cid):
        if self.alternative_X is not None:
            return np.asarray(self.alternative_X[f"{pid}-{cid}"])
        x = read_one_epoch(self.DATA_PATH, pid, cid)
        x = (x - self.percs[pid][0]) / (self.percs[pid][1] - self.percs[pid][0])
        return x

    def __getitem__(self, idx):
        pid = self.idx2pid(idx)
        curr_idx = idx - self._pid_to_start_cidx[pid]
        x = self.read_processed_data(pid, curr_idx)
        y = self.labels[idx]
        x = torch.FloatTensor(x).permute(1, 0)
        return idx, x, y

    def _get_perm(self, n):
        perm = np.random.permutation(n)
        return perm

    def get_split_indices(self, split_ratio, n, names=None):
        perm = self._get_perm(n)
        split_ratio = np.asarray(split_ratio).cumsum() / sum(split_ratio)
        cuts = [int(_s * n) for _s in split_ratio]
        if names is not None and len(names) == len(split_ratio):
            return {k: perm[cuts[i - 1]:cuts[i]] if i > 0 else perm[:cuts[0]] for i, k in enumerate(names)}

        return {'train': perm[:cuts[0]], 'val': perm[cuts[0]:]}

    @staticmethod
    def num_classes():
        return SHHS.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input1dSpec(
                seq_len=int(SHHS.MEASUREMENTS_PER_EXAMPLE), segment_size=SHHS.SEGMENT_SIZE, in_channels=SHHS.IN_CHANNELS
            ),
        ]


def train_val_test(root_folder, k, N, epoch_sec):
    all_index = sorted([int(path[6:12]) - 200000 for path in os.listdir(root_folder + 'shhs1')])

    train_index = np.random.choice(all_index, int(len(all_index) * 0.98), replace=False)
    test_index = np.random.choice(list(set(all_index) - set(train_index)), int(len(all_index) * 0.01), replace=False)
    val_index = list(set(all_index) - set(train_index) - set(test_index))

    sample_package(root_folder, k, N, epoch_sec, 'pretext', train_index)
    sample_package(root_folder, k, N, epoch_sec, 'train', test_index)
    sample_package(root_folder, k, N, epoch_sec, 'test', val_index)


#https://sleepdata.org/datasets/shhs/files/m/browser/documentation/SHHS1_Manual_of_Operations.pdf?inline=1
#EEG2 is C3 and at location 2, EEG is C4 at location 7


#Max = 0.000125, Min = -0.000125. Amplitude is 250 uV. Unit is V
def sample_package(root_folder, k, N, epoch_sec, dst_dir, index=None):
    if index is None:
        index = sorted([int(path[6:12]) - 200000 for path in os.listdir(os.path.join(root_folder, 'shhs1'))])

    for i, j in tqdm.tqdm(enumerate(index), total=len(index)):
        if i % N == k:
            curr_pid_dir = os.path.join(dst_dir, str(200000 + j))
            if not os.path.isdir(curr_pid_dir):
                os.makedirs(curr_pid_dir)
            if os.path.isfile(os.path.join(curr_pid_dir, 'label.pkl')):
                assert os.path.isfile(os.path.join(curr_pid_dir, 'percs.pkl'))
                continue

            # X load
            data = mne.io.read_raw_edf(os.path.join(root_folder, 'shhs1', f'shhs1-{str(200000 + j)}.edf'))
            X = data.get_data()
            if X.shape[0] == 16:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], :]
            elif X.shape[0] == 15:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14], :]
            X = X[[2, 7], :]

            data_chunks = pd.DataFrame(
                {
                    'both': np.percentile(X, [i for i in range(101)]),
                    'C3': np.percentile(X[0], [i for i in range(101)]),
                    'C4': np.percentile(X[1], [i for i in range(101)])
                }
            )
            pd.to_pickle(data_chunks, os.path.join(curr_pid_dir, 'percs.pkl'))

            # y load
            with open(os.path.join(root_folder, 'label', f'shhs1-{str(200000 + j)}-profusion.xml'), 'r') as infile:
                text = infile.read()
                root = ET.fromstring(text)
                y = [int(i.text) for i in root.find('SleepStages').findall('SleepStage')]

            for slice_index in range(X.shape[1] // (125 * epoch_sec)):
                path = os.path.join(curr_pid_dir, f"{slice_index}.npy")
                out = X[:, slice_index * 125 * epoch_sec:(slice_index + 1) * 125 * epoch_sec]
                np.save(path, out)
            else:
                pd.to_pickle(y[:slice_index + 1], os.path.join(curr_pid_dir, 'label.pkl'))


def get_ranges(root_folder, dst_dir):
    index = sorted([int(path[6:12]) - 200000 for path in os.listdir(os.path.join(root_folder, 'shhs1'))])
    res = {}
    for j in tqdm.tqdm(index):
        curr_pid_dir = os.path.join(dst_dir, str(200000 + j))
        percs = pd.read_pickle(os.path.join(curr_pid_dir, 'percs.pkl'))
        res[j] = percs.loc[100, 'both'] - percs.loc[0, 'both']
    return pd.Series(res)


def count_current_split():
    res = {}
    for split in ['pretext', 'train', 'test']:
        fnames = glob.glob(os.path.join(DATA_PATH, 'processed', split, "*.pkl"))
        for fname in tqdm.tqdm(fnames, desc=split):
            fname = os.path.basename(fname)
            _, pid, _ = fname.split("-")
            res[pid] = split
    return pd.Series(res)


def load_one_pid_label(datapath, pid):
    label = pd.Series(pd.read_pickle(os.path.join(datapath, 'processed', str(pid), 'label.pkl')))
    label[label == 5] = 4
    label[label > 5] = np.nan
    if len(label) > label.count():  # fill some corrupted data
        label = label.ffill()
        assert label.max() < 5 and label.count() == len(label)
        label = np.asarray(label).astype(int)
    else:
        label = label.values
    """
    2    1455937
    0    1029773
    4     525712
    3     419270
    1     133998
    9         88
    6          1
    """
    return label


def load_one_pid_perc(datapath, pid):
    percs = pd.read_pickle(os.path.join(datapath, 'processed', str(pid), 'percs.pkl'))
    return percs


def read_one_epoch(datapath, pid, cid):
    return np.load(os.path.join(datapath, 'processed', str(pid), f"{cid}.npy"))
