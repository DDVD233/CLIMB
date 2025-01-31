import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.specs import Input1dSpec

_MARGINAL_CLASS_DIST = {0: 15542, 1: 7629, 2: 18857, 3: 12046, 4: 8159}

DATASET = 'ISRUC'
ALL_CHANNELS = tuple(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
CHANNELS = tuple(['C3', 'C4'])
CLASSES = ['W', 'N1', 'N2', 'N3', 'R']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}  # R is called "5" in the original data, but use 4 here
CLASS_FREQ = {"W": 221, "N1": 126, "N2": 281, "N3": 214, "R": 62}  # Basing on patient_8_1
EPOCH_LENGTH = 30 * 125  # 30 seconds * 200 Hz but we need to resample to 125hz


class ISRUC(Dataset):
    '''A dataset class for the ISRUC SLEEP EEG dataset, The data were obtained from human adults, including healthy subjects,
     and subjects with sleep disorders under the effect of sleep medication. scoring sleep stages based on the AASM standard 5 stages
    (https://sleeptight.isr.uc.pt/)
    Note that you must go the EXTRACTED CHANNELS page, download the zip files from the 108 links to your base_root directory 
    then you must extract and rename to ISRUC_SLEEP, then you can run preprocess_isruc.py 
    to preprocess the data from mats to pkl and numpy files (30 second epochs)

    '''

    DATASET = DATASET
    LABEL_MAP = LABEL_MAP
    CLASS_FREQ = _MARGINAL_CLASS_DIST
    EPOCH_LENGTH = EPOCH_LENGTH  #30 * 150 but now resampled to 30*125
    CHANNELS = ['C3', 'C4']
    MEASUREMENTS_PER_EXAMPLE = 30 * 125
    NUM_CLASSES = 5  # 5 different clinical observations for classification
    SEGMENT_SIZE = 30
    IN_CHANNELS = 2  # 2-lead EEG readings
    RANDOM_SEED = 0
    TRAIN_SPLIT_FRAC = 0.8
    CLASSES = CLASSES

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:

        super().__init__()
        self.save_mem = False
        self.DATA_PATH = os.path.join(base_root, 'eeg', 'ISRUC_SLEEP')
        self.split = 'train' if train else 'val'
        split_ratios = [self.TRAIN_SPLIT_FRAC * 100, (1 - self.TRAIN_SPLIT_FRAC) * 100]
        patients = [int(x) for x in os.listdir(os.path.join(self.DATA_PATH, 'processed')) if x not in ['8', '18', 8, 18]]
        self.pids = self.get_split_indices(split_ratios, len(patients), names=['train', 'val'])[self.split]
        self.pids = sorted([patients[i] for i in self.pids])

        self.labels = []
        self._labels_2 = []
        self.groups = []
        self._pid_to_start_cidx = {}
        idx = 0
        for pid in self.pids:
            _ = load_one_pid_label(self.DATA_PATH, pid)
            _curr_label, _curr_label2 = _[0].values, _[1].values
            self._pid_to_start_cidx[pid] = idx
            for _cidx, y in enumerate(_curr_label):
                self.labels.append(y)
                self._labels_2.append(_curr_label2[_cidx])
                idx += 1
            self.groups.extend([pid] * len(_curr_label))
        self.labels = np.asarray(self.labels)
        self.groups = np.asarray(self.groups)
        self._labels_2 = np.asarray(self._labels_2)

        # normalizing constants
        self.percs = {}
        norm = None
        for pid in self.pids:
            if norm is None:
                self.percs[pid] = (-25, 25)
            else:
                _perc = load_one_pid_perc(self.DATA_PATH, pid)['all']
                assert isinstance(norm, int)
                self.percs[pid] = (_perc[norm], _perc[100 - norm])

        self.n = len(self.labels)
        self.alternative_X = None

        channels = self.CHANNELS

        if channels is None:
            channels = ALL_CHANNELS
        self.channel_locs = np.asarray([ALL_CHANNELS.index(x) for x in channels])
        self._mem = {}

    def __len__(self):
        return self.n

    def idx2pid(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.n:
            raise IndexError("%d is out of range (%d elements)" % (idx, self.n))
        return self.groups[idx]

    def get_raw_data(self, idx):
        raise NotImplementedError()
        import mne
        pid = self.idx2pid(idx)
        curr_idx = idx - (self.cumu_npoints_by_patients[pid] - self.npoints_by_pids[pid])
        EEG_raw = mne.io.read_raw_edf(os.path.join(self.DATA_PATH, f'{pid}/{pid}.edf'), preload=True)
        return EEG_raw, curr_idx, pid, self.actual_columns[pid]

    def read_processed_data(self, pid, cid):
        if self.save_mem:
            x = read_one_epoch(self.DATA_PATH, pid, cid)[self.channel_locs]
            x = (x - self.percs[pid][0]) / (self.percs[pid][1] - self.percs[pid][0])
        else:
            x = self._mem.get((pid, cid))

            if x is None:
                x = read_one_epoch(self.DATA_PATH, pid, cid)[self.channel_locs]
                x = (x - self.percs[pid][0]) / (self.percs[pid][1] - self.percs[pid][0])
                self._mem[(pid, cid)] = x
        return x

    def get_second_label(self, idx):
        return self._labels_2[idx]

    def __getitem__(self, idx):
        pid = self.idx2pid(idx)

        curr_idx = idx - self._pid_to_start_cidx[pid]
        #   print(pid, curr_idx)
        x = self.read_processed_data(pid, curr_idx)
        y = self.labels[idx]
        return idx, torch.FloatTensor(x).permute(1, 0), y

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
        return ISRUC.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input1dSpec(
                seq_len=int(ISRUC.MEASUREMENTS_PER_EXAMPLE), segment_size=ISRUC.SEGMENT_SIZE, in_channels=ISRUC.IN_CHANNELS
            ),
        ]


def find_channels(potential_channels, *, channels):
    # channels = ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1'] #https://arxiv.org/pdf/1910.06100.pdf

    keep = {}
    for c in potential_channels:
        new_c = c.replace("-M2", "").replace("-A2", "").replace("-M1", "").replace(
            "-A1", ""
        )  # https://www.ers-education.org/lrmedia/2016/pdf/298830.pdf
        if new_c in channels:
            assert new_c not in keep
            keep[new_c] = c
    assert len(keep) == len(channels), f"Something's wrong among columns={potential_channels}"
    return {v: k for k, v in keep.items()}


def load_one_mne(datapath, pid, channels):
    import mne
    EEG_raw_df = mne.io.read_raw_edf(os.path.join(datapath, f'{pid}/{pid}.edf')).copy().resample(sfreq=125).to_data_frame()
    try:
        rename_dict = find_channels(EEG_raw_df.columns, channels=channels)
    except Exception as err:
        print(pid, err)
        return None
    label1 = pd.read_csv(os.path.join(datapath, f"{pid}/{pid}_1.txt"), header=None)[0]
    label2 = pd.read_csv(os.path.join(datapath, f"{pid}/{pid}_2.txt"), header=None)[0]
    actual_data = EEG_raw_df.rename(columns=rename_dict).reindex(columns=channels)
    actual_columns = {v: k for k, v in rename_dict.items()}

    label1[label1 == 5] = 4
    label2[label2 == 5] = 4

    return label1, label2, actual_data, actual_columns


#Max = 25, Min = -25
def process_data_new(patient_ids=[1], *, datapath, channels, epoch_length):

    import time
    st = time.time()

    bad_pids = []
    for pid in tqdm.tqdm(patient_ids):
        dst_dir = os.path.join(datapath, 'processed', str(pid))
        label_cache_path = os.path.join(dst_dir, 'meta.pkl')
        if os.path.isfile(label_cache_path):
            assert len(pd.read_pickle(label_cache_path)) == 3
            continue

        #actual processing
        res = load_one_mne(os.path.join(datapath, 'raw'), pid, channels)
        if res is None:
            bad_pids.append(pid)
            print("bad pid: ", pid)
            continue
        label1, label2, actual_data, actual_columns = res

        assert len(actual_data) % epoch_length == 0
        n_epoch = int(len(actual_data) / epoch_length)
        assert n_epoch == len(label1)
        if n_epoch != len(label2):
            print(f"WARNING - Petient {pid}'s Label 2 is weird. Missing {n_epoch - len(label2)} / {n_epoch}")

        #curr_label, actual_columns = pd.read_pickle(label_cache_path)
        data_chunks = {'all': np.percentile(actual_data.values, [i for i in range(101)])}
        for k in actual_data.columns:
            data_chunks[k] = np.percentile(actual_data[k].values, [i for i in range(101)])
        data_chunks = pd.DataFrame(data_chunks)

        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        for curr_idx in range(n_epoch):

            curr_idx_cache_path = os.path.join(dst_dir, '%d.npy' % curr_idx)
            print(curr_idx_cache_path)
            if os.path.isfile(curr_idx_cache_path):
                continue
            x = actual_data.iloc[curr_idx * epoch_length:(curr_idx + 1) * epoch_length].values.T
            np.save(curr_idx_cache_path, x)
        else:
            curr_label = pd.DataFrame({0: label1, 1: label2})
            pd.to_pickle((curr_label, actual_columns, data_chunks), label_cache_path)

    print("Took %f seconds" % (time.time() - st))
    return bad_pids


def load_one_pid_label(datapath, pid):
    label, __, _ = pd.read_pickle(os.path.join(datapath, 'processed', str(pid), 'meta.pkl'))
    return label


def load_one_pid_perc(datapath, pid):
    __, _, percs = pd.read_pickle(os.path.join(datapath, 'processed', str(pid), 'meta.pkl'))
    return percs


def read_one_epoch(datapath, pid, cid):
    return np.load(os.path.join(datapath, 'processed', str(pid), f"{cid}.npy"))
