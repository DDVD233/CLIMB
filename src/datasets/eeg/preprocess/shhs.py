import glob
import os
import xml.etree.ElementTree as ET
from multiprocessing import Process

import mne
import numpy as np
import pandas as pd
import tqdm

DATASET = 'SHHS'
DATA_PATH = '/home/ubuntu/sleep/SHHS'
CHANNELS = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']  # https://arxiv.org/pdf/1910.06100.pdf
CLASSES = ['W', 'N1', 'N2', 'N3', 'R']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}  # R is called "5" in the oringinal data, but use 4 here
CLASS_FREQ = {"W": 221, "N1": 126, "N2": 281, "N3": 214, "R": 62}  # Basing on patient_8_1
EPOCH_LENGTH = 30 * 125  # 30 seconds * 125 Hz

_MIN = -0.000125
_MAX = 0.000125


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

            percentiles = pd.DataFrame(
                {
                    'both': np.percentile(X, [i for i in range(101)]),
                    'C3': np.percentile(X[0], [i for i in range(101)]),
                    'C4': np.percentile(X[1], [i for i in range(101)])
                }
            )
            pd.to_pickle(percentiles, os.path.join(curr_pid_dir, 'percs.pkl'))

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
    #    label, columns, percs = pd.read_pickle(os.path.join(datapath, 'processed', str(pid), 'meta.pkl'))
    return percs


def read_one_epoch(datapath, pid, cid):
    return np.load(os.path.join(datapath, 'processed', str(pid), f"{cid}.npy"))


if __name__ == '__main__':
    if not os.path.exists('./SHHS_data/processed/'):
        os.makedirs('./SHHS_data/processed/pretext')
        os.makedirs('./SHHS_data/processed/train')
        os.makedirs('./SHHS_data/processed/test')

    root_folder = DATA_PATH
    result_dir = os.path.join(root_folder, 'processed')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    all_index = sorted([int(path[6:12]) - 200000 for path in os.listdir(os.path.join(root_folder, 'shhs1'))])
    sample_package(root_folder, 0, 1, 30, result_dir)
    num_processes, epoch_sec = 30, 30
    res = get_ranges(root_folder, result_dir)
    p_list = []
    for k in range(num_processes):
        process = Process(target=sample_package, args=(root_folder, k, num_processes, epoch_sec, result_dir))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()
