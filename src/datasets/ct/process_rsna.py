"""Preprocessing script for the RSNA datset. 

Before running the script, visit https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection to download the data
Once the download completes, place the zip file in './src/datasets/ct/rsna' and unzip the file

The Preprocessing script includes the following steps: 
    - Extract DICOM metadata for each CT exam. 
    - Process and store pixel array from DICOM files to HDF5 format for faster output speed. 
    - Convert each CT series to sliding windows of N slices.
"""

import os
import pickle
import sys
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import pydicom
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.getcwd())
from src.datasets.ct.dicom_utils import read_dicom
# constant variables for the RSNA dataset
from src.datasets.ct.rsna import *

RANDOM_SEED = 6


def process_window_df(
    df: pd.DataFrame,
    num_slices: int = RSNA_WINDOW_SIZE,
    min_abnormal_slice: int = 4,
    stride: int = None,
):
    f"""
    Convert each CT series to sliding windows of N slices. The processed window
    level information are stored in {RSNA_WINDOWS_CSV}.
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
    count_num_windows = lambda x: (x - num_slices) // stride
    df[RSNA_NUM_WINDOW_COL] = df[RSNA_NUM_SLICES_COL].apply(count_num_windows)

    # get windows list
    df_study = df.groupby([RSNA_STUDY_COL]).head(1)
    window_labels = defaultdict(list)

    # studies
    for _, row in tqdm.tqdm(df_study.iterrows(), total=df_study.shape[0]):
        study_name = row[RSNA_STUDY_COL]
        split = row[RSNA_SPLIT_COL]

        study_df = df[df[RSNA_STUDY_COL] == study_name]
        study_df = study_df.sort_values(RSNA_PATIENT_POSITION_COL)

        # windows
        for idx in range(row[RSNA_NUM_WINDOW_COL]):
            start_idx = idx * stride
            end_idx = (idx * stride) + num_slices

            window_df = study_df.iloc[start_idx:end_idx]

            # central PEs
            num_central_positives_slices = window_df[RSNA_CENTRAL_PE_COL].sum()
            central_label = (1 if num_central_positives_slices >= min_abnormal_slice else 0)

            # non-central positive PEs
            num_non_central_positives_slices = window_df[window_df[RSNA_CENTRAL_PE_COL] == 0][RSNA_PE_SLICE_COL].sum()
            non_central_label = (1 if num_non_central_positives_slices >= min_abnormal_slice else 0)

            # any pe positive label
            label = 1 if (central_label == 1 or non_central_label == 1) else 0

            window_labels[RSNA_STUDY_COL].append(study_name)
            window_labels[RSNA_WINDOW_INDEX_COL].append(idx)
            window_labels[RSNA_WINDOW_LABEL_COL].append(label)
            window_labels[RSNA_WINDOW_CENTRAL_LABEL_COL].append(central_label)
            window_labels[RSNA_WINDOW_NON_CENTRAL_LABEL_COL].append(non_central_label)
            window_labels[RSNA_SPLIT_COL].append(split)
            window_labels[RSNA_INSTANCE_ORDER_COL].append(window_df[RSNA_INSTANCE_ORDER_COL].tolist())

    df = pd.DataFrame.from_dict(window_labels)
    df.to_csv(RSNA_WINDOWS_CSV)

    return df


def process_study_to_hdf5(csv_path: str = RSNA_TRAIN_CSV, hdf5_path: str = RSNA_STUDY_HDF5):
    """Save DICOM study to hdf5 file

    Args:
        csv_path (str): path to csv file that contains information for the
            series to be stored in HDF5
        hdf5_path (str): path to the hdf5 file for storing CT series
    """

    df = pd.read_csv(csv_path)
    hdf5_fh = h5py.File(hdf5_path, "a")

    all_studies = df[RSNA_STUDY_COL].unique()

    for study_name in tqdm.tqdm(all_studies, total=len(all_studies)):
        study_df = df[df[RSNA_STUDY_COL] == study_name].copy()

        # order study instances
        study_df = study_df.sort_values(RSNA_PATIENT_POSITION_COL)

        # save paths to hdf5
        instance_paths = study_df[RSNA_INSTANCE_PATH_COL].tolist()
        series = np.stack([read_dicom(RSNA_TRAIN_DIR / path, RSNA_IMAGE_SIZE) for path in instance_paths])
        hdf5_fh.create_dataset(study_name, data=series, dtype="float32", chunks=True)

    # clean up
    hdf5_fh.close()


class DICOMMetadata(Dataset):
    """PyTorch Dataset for parallelizing DICOM headers extraction

    Attributes:
        df (pd.DataFrame): DataFrame with path to DICOM files
    """

    def __init__(self, df: pd.DataFrame):

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        dcm = pydicom.dcmread(RSNA_TRAIN_DIR / row[RSNA_INSTANCE_PATH_COL], stop_before_pixels=True)

        metadata = {}
        for k in RSNA_DICOM_HEADERS:
            try:
                att = getattr(dcm, k)

                if k in DICOM_NUMERIC_HEADERS:
                    metadata[k] = float(att)
                elif k in DICOM_SEQUENCE_HEADERS:
                    for ind, coord in enumerate(att):
                        metadata[f"{k}_{ind}"] = float(coord)
                else:
                    metadata[k] = str(att)
            except Exception as e:
                print(e)

        return pd.DataFrame(metadata, index=[0])


def add_split_to_label_df(
    label_df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    study_col: str = RSNA_STUDY_COL,
    split_col: str = RSNA_SPLIT_COL,
):
    """Create train/val/test split for a given dataframe.

    Splits are made based on patients instead of studies. This is to ensure that
    studies from the same patients do not appear in two different splits.

    Args:
        label_df (pd.DataFrame): Dataframe to add split information to.
        val_size (float): Percentage of data in validation set.
        test_size (float): Percentage of data in test set.
        study_col (str): Study column name used to split data.
        split_col (str): Column used to indicate split.

    Returns:
        DataFrame with a new Split column, indicating the split for each study.
    """
    patients = label_df[study_col].unique()

    # split between train and val+test
    split_ratio = val_size + test_size
    train_patients, test_val_patients = train_test_split(patients, test_size=split_ratio, random_state=RANDOM_SEED)

    # split between val and test
    test_split_ratio = test_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(test_val_patients, test_size=test_split_ratio, random_state=RANDOM_SEED)
    train_rows = label_df[study_col].isin(train_patients)
    label_df.loc[train_rows, split_col] = "train"
    val_rows = label_df[study_col].isin(val_patients)
    label_df.loc[val_rows, split_col] = "valid"
    test_rows = label_df[study_col].isin(test_patients)
    label_df.loc[test_rows, split_col] = "test"

    return label_df


def extract_dicom_metadata(df: pd.DataFrame):
    """Extract DICOM metadata

    Args:
        df (pd.DataFrame): DataFrame that contains paths to each CT slice for
            which metatdata will be extracted.

    Returns:
        The input dataframe with extract metadata.
    """

    # create dataset and loader to extract metadata
    dataset = DICOMMetadata(df)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=lambda x: x)

    # get metadata
    meta = []
    for data in tqdm.tqdm(loader, total=len(loader)):
        meta += [data[0]]
    meta_df = pd.concat(meta, axis=0, ignore_index=True)

    # get slice number
    unique_studies = pd.DataFrame(meta_df[RSNA_STUDY_COL].value_counts()).reset_index()
    unique_studies.columns = [RSNA_STUDY_COL, RSNA_NUM_SLICES_COL]
    meta_df = meta_df.merge(unique_studies, on=RSNA_STUDY_COL)

    return meta_df


def get_slice_neighbors(df):
    """Sort CT studies by patient position and indicate neighboring slices.

    ImageNet pretrained models except inputs to have 3 channels (RGB). However,
    CT scans are grayscale, so preprocessing are required to adept them to 3
    channels. One stretegy is to pad a slice with its neighboring slices to
    provide contextual information. This function sorts each CT series by its
    patient position and indicates its neighboring slices.

    Args:
        df (pd.DataFrame): DataFrame that contains slices that requires sorting.

    Returns:
        The same DataFrame as input with slice order and paths to neighboring
        slices.
    """

    study_dfs = []
    for study_name in tqdm.tqdm(df[RSNA_STUDY_COL].unique(), total=df[RSNA_STUDY_COL].nunique()):
        study_df = df[df[RSNA_STUDY_COL] == study_name].copy()

        # order study instances
        study_df = study_df.sort_values(RSNA_PATIENT_POSITION_COL)
        study_df[RSNA_INSTANCE_ORDER_COL] = np.arange(len(study_df))

        # get neighbors paths
        instance_paths = study_df[RSNA_INSTANCE_PATH_COL].tolist()
        instance_paths = [instance_paths[0]] + instance_paths + [instance_paths[-1]]
        study_df[RSNA_PREV_INSTANCE_COL] = instance_paths[:-2]
        study_df[RSNA_NEXT_INSTANCE_COL] = instance_paths[2:]

        study_dfs.append(study_df)

    df = pd.concat(study_dfs, axis=0, ignore_index=True)

    return df


def preprocess_rsna():
    """Preprocess steps for the RSNA PE dataset, including:
    - indicate Stanford studies.
    - create train/val/test/stanford split.
    - extract DICOM metadata.
    - indicate neighboring slices.
    """

    # applying train/val/test split only on train csv (test.csv does not contain labels)
    df = pd.read_csv(RSNA_ORIGINAL_TRAIN_CSV)

    # split stanford studies
    if not os.path.exists(STANFORD_STUDIES_IN_RSNA):
        raise RuntimeError(
            """
            Download stanford_study_in_rsna.pkl file from the link below by 
            clicking on the 'Download' button on the top right corner of the 
            screen:
                https://stanfordmedicine.box.com/s/e8egudko81608yyhb2zxy2urcpj73e3e
            

            Once the file is downloaded, please move the file to: 
                ./src/datasets/ct/rspect/
            """
        )
    stanford_studies = pickle.load(open(STANFORD_STUDIES_IN_RSNA, "rb"))
    df_stanford = df[df[RSNA_STUDY_COL].isin(stanford_studies)]
    df_others = df[~df[RSNA_STUDY_COL].isin(stanford_studies)]

    # full dataset split
    df_others = add_split_to_label_df(df_others, split_col=RSNA_SPLIT_COL)
    df_stanford.loc[:, RSNA_SPLIT_COL] = "external"
    df = pd.concat([df_others, df_stanford])

    # add instance path
    df[RSNA_INSTANCE_PATH_COL] = df.apply(
        lambda x: f"{x[RSNA_STUDY_COL]}/{x[RSNA_SERIES_COL]}/{x[RSNA_INSTANCE_COL]}.dcm",
        axis=1,
    )

    # get metadata
    meta_df = extract_dicom_metadata(df)
    df = df.set_index([RSNA_STUDY_COL, RSNA_SERIES_COL, RSNA_INSTANCE_COL])
    meta_df = meta_df.set_index([RSNA_STUDY_COL, RSNA_SERIES_COL, RSNA_INSTANCE_COL])
    df = df.join(meta_df, how="left").reset_index()

    # sort study by slices order and get neighboring slices information
    df = get_slice_neighbors(df)

    # save csv file
    df.to_csv(RSNA_TRAIN_CSV, index=False)


def print_log(log: str):
    """Helper function to style output log"""
    print("=" * 80)
    print(log)
    print("-" * 80)


if __name__ == "__main__":

    # preprocess rsna datasets and extract DICOM metadata
    if not RSNA_TRAIN_CSV.is_file():
        print_log(f"\nProcessing RSNA dataset metadata and saving as {RSNA_TRAIN_CSV}")
        preprocess_rsna()
    else:
        print_log(f"\n{RSNA_TRAIN_CSV} already existed and processed")

    # process HDF5
    if not RSNA_STUDY_HDF5.is_file():
        print_log(f"\nParsing study HDF5 to {RSNA_STUDY_HDF5}")
        process_study_to_hdf5(RSNA_TRAIN_CSV)
    else:
        print_log(f"\n{RSNA_STUDY_HDF5} already existed and processed")

    # process CT windows
    if not RSNA_WINDOWS_CSV.is_file():
        print_log(f"\nParsing windows csv to {RSNA_WINDOWS_CSV}")
        df = pd.read_csv(RSNA_TRAIN_CSV)
        process_window_df(df, num_slices=RSNA_WINDOW_SIZE, min_abnormal_slice=4, stride=12)
    else:
        print_log(f"\n{RSNA_WINDOWS_CSV} already existed and processed")
