import os

import numpy as np
import pandas as pd
import tqdm

import utils

DATASET = 'ISRUC'
DATA_PATH = '/home/dvd/Datasets/high_modality/eeg/ISRUC_SLEEP'
ALL_CHANNELS = tuple(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
CHANNELS = tuple(['C3', 'C4'])
CLASSES = ['W', 'N1', 'N2', 'N3', 'R']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}  # R is called "5" in the oringinal data, but use 4 here
CLASS_FREQ = {"W": 221, "N1": 126, "N2": 281, "N3": 214, "R": 62}  # Basing on patient_8_1
EPOCH_LENGTH = 30 * 125  # 30 seconds * 200 Hz but we need to resample to 125hz


def download_file(url, dst):
    import requests
    r = requests.get(url, stream=True)
    with open(dst, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def download_ISRUC(group=1):
    import pyunpack
    assert group == 1, "I only checked group 1 data"
    data_dir = DATA_PATH
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    #https://sleeptight.isr.uc.pt/?page_id=48
    for patient_id in range(1, 100 + 1):  #patient 1 to 100
        if os.path.isfile(os.path.join(data_dir, '%d/%d.edf' % (patient_id, patient_id))):
            continue

        rar_url = "http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/%d.rar" % patient_id
        rar_dst = os.path.join(data_dir, "%d.rar" % patient_id)
        if not os.path.isfile(rar_dst):
            download_file(rar_url, rar_dst)

        #unrar
        unzipped_dst = os.path.join(data_dir, "")
        #patoolib.extract_archive(rar_dst, outdir=unzipped_dst)
        pyunpack.Archive(rar_dst).extractall(unzipped_dst)
        #rename
        os.rename(
            os.path.join(data_dir, '%d/%d.rec' % (patient_id, patient_id)),
            os.path.join(data_dir, '%d/%d.edf' % (patient_id, patient_id))
        )

        #Delete the rar
        os.remove(rar_dst)


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


#import functools
#@functools.lru_cache()
def load_one_mne(datapath, pid, channels):
    import mne
    EEG_raw_df = mne.io.read_raw_edf(os.path.join(datapath, f'{pid}/{pid}.edf'), preload=True)
    EEG_raw_df = EEG_raw_df.resample(sfreq=125).to_data_frame()
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
        percentiles = {'all': np.percentile(actual_data.values, [i for i in range(101)])}
        for k in actual_data.columns:
            percentiles[k] = np.percentile(actual_data[k].values, [i for i in range(101)])
        percentiles = pd.DataFrame(percentiles)

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
            pd.to_pickle((curr_label, actual_columns, percentiles), label_cache_path)

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


#================================================DEBUG


def sep_cache_and_raw(dir=DATA_PATH):
    import shutil
    dst_meta_dir = os.path.join(dir, 'raw')
    if not os.path.isdir(dst_meta_dir):
        os.makedirs(dst_meta_dir)
    for i in tqdm.tqdm(range(1, 101)):
        curr_dir = os.path.join(dst_meta_dir, str(i))
        if not os.path.isdir(curr_dir):
            os.makedirs(curr_dir)
        for f in os.listdir(os.path.join(dir, str(i))):
            fpath = os.path.join(dir, str(i), f)
            if not os.path.isdir(fpath):
                assert os.path.isfile(fpath)
                shutil.move(fpath, os.path.join(curr_dir, os.path.basename(fpath)))


def get_ranges(datapath):
    res = {}
    for pid in range(1, 101):
        if pid == 8:
            continue
        __, _, percs = pd.read_pickle(os.path.join(datapath, 'processed', str(pid), 'meta.pkl'))
        res[pid] = percs.loc[100, 'all'] - percs.loc[0, 'all']
    return pd.Series(res)


if __name__ == "__main__":
    # download_ISRUC()
    # sep_cache_and_raw()
    # task_runner = utils.TaskPartitioner()
    # for i in range(1, 101):
    #     task_runner.add_task(
    #         process_data_new, tuple([i]), datapath=DATA_PATH, epoch_length=EPOCH_LENGTH, channels=ALL_CHANNELS
    #     )
    # res = task_runner.run_multi_process(4, cache_only=False)
    for i in range(1, 101):
        process_data_new([i], datapath=DATA_PATH, channels=ALL_CHANNELS, epoch_length=EPOCH_LENGTH)
# res = load_data(channels=['C3', 'C4'], datapath=DATA_PATH, epoch_length=EPOCH_LENGTH)
