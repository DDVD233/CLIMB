import os
from ast import literal_eval
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm

from src.datasets.specs import Input3dSpec

from .base_ct import CTDatasetBase

# RadFusion
RADFUSION_DATA_DIR = Path("/home/ubuntu/radfusion/multimodalpulmonaryembolismdataset")
if not RADFUSION_DATA_DIR.is_dir():
    RADFUSION_DATA_DIR.mkdir(parents=True, exist_ok=False)

RADFUSION_ORIGINAL_CSV = RADFUSION_DATA_DIR / "Labels.csv"
RADFUSION_IMG_DIRS = [RADFUSION_DATA_DIR / str(i) for i in range(8)]
RADFUSION_MASTER_CSV = RADFUSION_DATA_DIR / "master.csv"
RADFUSION_WINDOWS_CSV = RADFUSION_DATA_DIR / "window.csv"
RADFUSION_HDF5 = RADFUSION_DATA_DIR / "data.hdf5"
RADFUSION_SPLIT_COL = "split"
RADFUSION_STUDY_COL = "idx"
RADFUSION_PATH_COL = "path"
RADFUSION_SLICE_IDX_COL = "slice_idx"
RADFUSION_NUM_SLICES_COL = "num_slices"
RADFUSION_PE_LABEL_COL = "label"
RADFUSION_PE_TYPE_COL = "pe_type"
RADFUSION_SPLI_COL = "split"
RADFUSION_NUM_WINDOW_COL = 'num_window'
RADFUSION_WINDOW_INDEX_COL = 'window_index'
RADFUSION_WINDOW_LABEL_COL = 'window_label'
RADFUSION_WINDOW_CENTRAL_LABEL_COL = 'window_central_label'
RADFUSION_WINDOW_NON_CENTRAL_LABEL_COL = 'window_non_central_label'
RADFUSION_INSTANCE_ORDER_COL = 'instance_order'

RADFUSION_WINDOW_SIZE = 24
RADFUSION_WINDOW_STRIDE = 12
RADFUSION_MIN_ABNORMAL_SLICE = 4
RADFUSION_IMAGE_SIZE = 256

RADFUSION_INPUT_SIZE = (RADFUSION_IMAGE_SIZE, RADFUSION_IMAGE_SIZE, RADFUSION_WINDOW_SIZE)
RADFUSION_PATCH_SIZE = (16, 16, RADFUSION_WINDOW_SIZE)
RADFUSION_IN_CHANNELS = 1
PE_WINDOW_CENTER = 400
PE_WINDOW_WIDTH = 1000


def any_exist(files):
    return any(map(os.path.exists, files))


def all_exist(files):
    return all(map(os.path.exists, files))


class RadfusionDatasetWindow(CTDatasetBase):
    """A dataset class for the RadFusion dataset: a multi-modal benchmark dataset 
    of 1794 patients with corresponding EHR data and high-resolution computed 
    tomography (CT) scans labeled for pulmonary embolism from Stanford Healthcare Center. 

    In addition to labels for PE positive/negative, the dataset also includes
    labels the type of PE (central, subsegmental, segemental) and summarized 
    electronic health record (i.e. medication, lab test value, ICD, vitals)

    We categorize PE labels into 3 categories:
        - Central PE (based on a window having at least 4 slices with central PE)
        - Other PE (based on a window having at least 4 slices of non-central PE).
            Non-central PE are categorized by either segmental or subsegmental 
            PE label.

    While we train the model using windows of slices instead of the full series,
    the final performance is calculated by aggregating prediction probabilities,
    by taking the maximum value, from all windows in a series.

    More information can be found in the full manuscript:
        https://arxiv.org/abs/2111.11665
    """

    def __init__(self, base_root: str = RADFUSION_DATA_DIR, download: bool = True, train: bool = True):

        # check if RadFusion is downloaded and pre-processed
        self.check_data()

        # literal_eval to convert str to list
        self.df = pd.read_csv(RADFUSION_WINDOWS_CSV,)

        if train:
            self.df = self.df[self.df[RADFUSION_SPLIT_COL] == "train"]
        else:
            self.df = self.df[self.df[RADFUSION_SPLIT_COL] == "test"]

        super().__init__(train)

    def check_data(self):

        # if no data is present, prompt the user to download it
        if not any_exist([RADFUSION_ORIGINAL_CSV] + RADFUSION_IMG_DIRS):
            raise RuntimeError(
                f"""
                Visit https://stanfordaimi.azurewebsites.net/datasets/3a7548a4-8f65-4ab7-85fa-3d68c9efc1bd to download the RadFusion dataset
                Once the download completes, place the zip file in {RADFUSION_DATA_DIR} and unzip the file
                """
            )

        # if the data has not been extracted, extract the data, prioritizing the full-res dataset
        if not all_exist([RADFUSION_MASTER_CSV, RADFUSION_WINDOWS_CSV, RADFUSION_HDF5]):
            print("Pre-proessing RadFusion...")
            self.preprocess_data()
            print("RadFusion prep-rocessing complete!")

    def __getitem__(self, index):

        row = self.df.iloc[index]
        study_name = row[RADFUSION_STUDY_COL]
        slice_idx = sorted(literal_eval(row[RADFUSION_INSTANCE_ORDER_COL]))

        # extract and transform window
        window = self.read_from_hdf5(hdf5_path=RADFUSION_HDF5, key=study_name, slice_idx=slice_idx)
        window = self.windowing(window, PE_WINDOW_CENTER, PE_WINDOW_WIDTH)
        if len(window.shape) == 3:
            window = np.expand_dims(window, 1)

        x = torch.from_numpy(window).float()
        x = x.permute((1, 2, 3, 0))
        x = x.type(torch.FloatTensor)

        # get labels
        y = torch.tensor(row[RADFUSION_WINDOW_LABEL_COL]).type(torch.LongTensor)
        print("x", x.shape, "y", y.item())
        return index, x, y.item()

    def __len__(self):
        return len(self.df)

    @staticmethod
    def num_classes():
        """Predicting Central Positive and Non-central Positive"""
        return 1

    @staticmethod
    def spec():
        return [
            Input3dSpec(input_size=RADFUSION_INPUT_SIZE, patch_size=RADFUSION_PATCH_SIZE, in_channels=1),
        ]

    def preprocess_data(self):

        df = pd.read_csv(RADFUSION_ORIGINAL_CSV)

        # store series information
        idx_2_path = {}
        idx_2_slices = {}

        # read series and store in HDF5
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        hdf5_fh = h5py.File(RADFUSION_HDF5, "w")
        paths = [p for d in RADFUSION_IMG_DIRS for p in d.glob('*.npy')]
        for p in tqdm.tqdm(paths, total=len(paths)):

            # get series name from path
            idx = str(p).split('/')[-1].split('.')[0]

            # store path information
            idx_2_path[idx] = p

            # store arr in hdf5
            img_stack = np.load(p)

            width = RADFUSION_IMAGE_SIZE

            height = RADFUSION_IMAGE_SIZE
            img_stack_sm = np.zeros((len(img_stack), width, height))

            for iidx in range(len(img_stack)):
                img = img_stack[iidx, :, :]
                img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img_stack_sm[iidx, :, :] = img_sm
            hdf5_fh.create_dataset(str(idx), data=img_stack_sm, dtype="int16", chunks=True)

            # store number of slices
            num_slices = img_stack_sm.shape[0]
            idx_2_slices[idx] = num_slices

        # clean up
        hdf5_fh.close()

        # append series information to dataframe
        df[RADFUSION_PATH_COL] = df[RADFUSION_STUDY_COL].apply(lambda x: idx_2_path[str(x)])
        df[RADFUSION_NUM_SLICES_COL] = df[RADFUSION_STUDY_COL].apply(lambda x: idx_2_slices[str(x)])

        # save new master csv
        df.to_csv(RADFUSION_MASTER_CSV)

        # window df
        df_window = self.process_window_df(df)
        df_window.to_csv(RADFUSION_WINDOWS_CSV)

    def process_window_df(
        self,
        df: pd.DataFrame,
        num_slices: int = RADFUSION_WINDOW_SIZE,
        min_abnormal_slice: int = RADFUSION_MIN_ABNORMAL_SLICE,
        stride: int = RADFUSION_WINDOW_STRIDE,
    ):
        f"""
        Convert each CT series to sliding windows of N slices. The processed window
        level information are stored in {RADFUSION_WINDOWS_CSV}.
        The number of windows for a series is calculated by:
            (num_windows - num_slices) // stride
        Args:
            df (pd.DataFrame): DataFrame with series level information
            num_slices (int); number of slices per window
            min_abnormal_slice (int): number of abnormal slices needed to consider a
                window as abnormal
            stride (int): spacing between each window

        Returns:
            A DataFrame with processed window level information.
        """

        # count number of windows per series
        def count_num_windows(x):
            return (x - num_slices) // stride

        df[RADFUSION_NUM_WINDOW_COL] = df[RADFUSION_NUM_SLICES_COL].apply(count_num_windows)

        # get windows list
        df_study = df.groupby([RADFUSION_STUDY_COL]).head(1)
        window_labels = defaultdict(list)

        # studies
        for _, row in tqdm.tqdm(df_study.iterrows(), total=df_study.shape[0]):
            study_name = row[RADFUSION_STUDY_COL]
            split = row[RADFUSION_SPLIT_COL]

            study_df = df[df[RADFUSION_STUDY_COL] == study_name]

            # windows
            for idx in range(row[RADFUSION_NUM_WINDOW_COL]):
                start_idx = idx * stride
                end_idx = (idx * stride) + num_slices

                window_df = study_df.iloc[start_idx:end_idx]

                pe_type_count = Counter(window_df[RADFUSION_PE_TYPE_COL].tolist())

                # central PEs
                num_central_positives_slices = pe_type_count['central']
                central_label = (1 if num_central_positives_slices >= min_abnormal_slice else 0)

                # non-central positive PEs (segmental + subsegmental)
                num_non_central_positives_slices = pe_type_count['segmental'] + \
                    pe_type_count['subsegmental']
                non_central_label = (1 if num_non_central_positives_slices >= min_abnormal_slice else 0)

                # any pe positive label
                label = 1 if 1 in set([row[RADFUSION_PE_LABEL_COL]]) else 0

                window_labels[RADFUSION_STUDY_COL].append(study_name)
                window_labels[RADFUSION_SPLIT_COL].append(split)
                window_labels[RADFUSION_WINDOW_INDEX_COL].append(idx)
                window_labels[RADFUSION_WINDOW_LABEL_COL].append(label)
                window_labels[RADFUSION_WINDOW_CENTRAL_LABEL_COL].append(central_label)
                window_labels[RADFUSION_WINDOW_NON_CENTRAL_LABEL_COL].append(non_central_label)
                window_labels[RADFUSION_INSTANCE_ORDER_COL].append(np.arange(start_idx, end_idx).tolist())
        df = pd.DataFrame.from_dict(window_labels)

        return df
