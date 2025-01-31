import os
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.datasets.specs import Input3dSpec

from .base_ct import CTDatasetBase

# RSNA
RSNA_DATA_DIR = Path("./src/datasets/ct/rspect")
if not RSNA_DATA_DIR.is_dir():
    RSNA_DATA_DIR.mkdir(parents=True, exist_ok=False)

RSNA_ORIGINAL_TRAIN_CSV = RSNA_DATA_DIR / "train.csv"
RSNA_TRAIN_CSV = RSNA_DATA_DIR / "rsna_master.csv"
RSNA_WINDOWS_CSV = RSNA_DATA_DIR / "rsna_windows.csv"
RSNA_STUDY_HDF5 = RSNA_DATA_DIR / "rsna_study.hdf5"
RSNA_TRAIN_DIR = RSNA_DATA_DIR / "train"

STANFORD_STUDIES_IN_RSNA = RSNA_DATA_DIR / "stanford_study_in_rsna.pkl"

RSNA_STUDY_COL = "StudyInstanceUID"
RSNA_SERIES_COL = "SeriesInstanceUID"
RSNA_INSTANCE_COL = "SOPInstanceUID"
RSNA_PREV_INSTANCE_COL = "PrevSOPInstanceUID"
RSNA_NEXT_INSTANCE_COL = "NextSOPInstanceUID"
RSNA_INSTANCE_PATH_COL = "InstancePath"
RSNA_SPLIT_COL = "Split"
RSNA_INSTITUTION_COL = "Institution"
RSNA_INSTANCE_ORDER_COL = "InstanceOrder"
RSNA_PE_SLICE_COL = "pe_present_on_image"
RSNA_CENTRAL_PE_COL = "central_pe"
RSNA_PATIENT_POSITION_COL = "ImagePositionPatient_2"
RSNA_NUM_SLICES_COL = "NumSlices"
RSNA_NUM_WINDOW_COL = "NumWindows"
RSNA_WINDOW_INDEX_COL = "WindowIdx"
RSNA_WINDOW_LABEL_COL = "WindowLabel"
RSNA_WINDOW_CENTRAL_LABEL_COL = "WindowCentralPELabel"
RSNA_WINDOW_NON_CENTRAL_LABEL_COL = "WindowNonCentralPELabel"
RSNA_WINDOW_SIZE = 24
RSNA_IMAGE_SIZE = 256
RSNA_CROP_SIZE = 224

DICOM_NUMERIC_HEADERS = ["InstanceNumber", "RescaleSlope", "RescaleIntercept"]
DICOM_SEQUENCE_HEADERS = [
    "PixelSpacing",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]
DICOM_STRING_HEADERS = [
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "WindowCenter",
    "WindowWidth",
]
RSNA_DICOM_HEADERS = (DICOM_NUMERIC_HEADERS + DICOM_SEQUENCE_HEADERS + DICOM_STRING_HEADERS)

RSNA_INPUT_SIZE = (RSNA_IMAGE_SIZE, RSNA_IMAGE_SIZE, RSNA_WINDOW_SIZE)
RSNA_PATCH_SIZE = (16, 16, RSNA_WINDOW_SIZE)
RSNA_IN_CHANNELS = 1
PE_WINDOW_CENTER = 400
PE_WINDOW_WIDTH = 1000


def any_exist(files):
    return any(map(os.path.exists, files))


class RSNADatasetWindow(CTDatasetBase):
    """A dataset class for the RSNA PE dataset (RSPECT): This multi-instituional,
    multi-national dataset is the largest publicly available pulmonary embolism
    (PE) CT dataset as of 2022. The dataset includes both series-level and 
    slice-level annotations for PE, from a large group of subspecialist thoracic 
    radiologists.

    In addition to labels for PE positive/negative, the dataset also includes
    labels indicating the side of PE (left vs right) and the type of PE (chronic,
    central, ect).

    We categorize PE labels into 2 categories:
        - Central PE (based on a window having at least 4 slices with central PE)
        - Other PE (based on a window having at least 4 slices of non-central PE).
            Non-central PE are categorized by any PE positive slices without the
            central PE label.

    While we train the model using windows of slices instead of the full series,
    the final performance is calculated by aggregating prediction probabilities,
    by taking the maximum value, from all windows in a series.

    More information can be found in the full manuscript:
        https://pubs.rsna.org/doi/full/10.1148/ryai.2021200254
    """

    def __init__(self, base_root: str = RSNA_DATA_DIR, download: bool = True, train: bool = True):
        super().__init__(train)

        # check if RSNA PE dataset is downloaded and pre-processed
        self.check_data()

        # literal_eval to convert str to list
        self.df = pd.read_csv(
            RSNA_WINDOWS_CSV,
            converters={
                RSNA_INSTANCE_PATH_COL: literal_eval,
                RSNA_PE_SLICE_COL: literal_eval,
                RSNA_INSTANCE_COL: literal_eval,
                RSNA_INSTANCE_ORDER_COL: literal_eval,
            },
        )

        if train:
            self.df = self.df[self.df[RSNA_SPLIT_COL] == "train"]
        else:
            self.df = self.df[self.df[RSNA_SPLIT_COL] == "valid"]

    def check_data(self):
        RSNA_WINDOWS_CVS = RSNA_DATA_DIR / "rsna_windows.csv"
        RSNA_STUDY_HDF5 = RSNA_DATA_DIR / "rsna_study.hdf5"

        if not os.path.exists(STANFORD_STUDIES_IN_RSNA):
            raise RuntimeError(
                f"""
                Download stanford_study_in_rsna.pkl file from the link below by 
                clicking on the 'Download' button on the top right corner of the 
                screen:
                    https://stanfordmedicine.box.com/s/e8egudko81608yyhb2zxy2urcpj73e3e
                

                Once the file is downloaded, please move the file to: 
                    {RSNA_DATA_DIR}
                """
            )

        # if no data is present, prompt the user to download it
        if not any_exist([RSNA_ORIGINAL_TRAIN_CSV, RSNA_TRAIN_DIR]):
            raise RuntimeError(
                f"""
                Visit https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection to download the data
                Once the download completes, place the zip file in {RSNA_DATA_DIR} and unzip the file
                """
            )

        # if the data has not been extracted, extract the data, prioritizing the full-res dataset
        if not any_exist([RSNA_TRAIN_CSV, RSNA_WINDOWS_CVS, RSNA_STUDY_HDF5]):
            raise RuntimeError(
                """
                Preprocess RSNA dataset using:: 
                    $ python src/datasets/ct/process_rsna.py
                """
            )

    def __getitem__(self, index):

        row = self.df.iloc[index]
        study_name = row[RSNA_STUDY_COL]
        slice_idx = sorted(row[RSNA_INSTANCE_ORDER_COL])

        # extract and transform window
        window = self.read_from_hdf5(hdf5_path=RSNA_STUDY_HDF5, key=study_name, slice_idx=slice_idx)
        window = self.windowing(window, PE_WINDOW_CENTER, PE_WINDOW_WIDTH)
        if len(window.shape) == 3:
            window = np.expand_dims(window, 1)

        x = torch.from_numpy(window).float()
        x = x.permute((1, 2, 3, 0))
        x = x.type(torch.FloatTensor)

        # get labels
        y = torch.tensor(row[RSNA_WINDOW_LABEL_COL]).type(torch.LongTensor)

        return study_name, x, y

    def __len__(self):
        return len(self.df)

    @staticmethod
    def num_classes():
        """Predicting Central Positive and Non-central Positive"""
        return 1

    @staticmethod
    def spec():
        return [
            Input3dSpec(input_size=RSNA_INPUT_SIZE, patch_size=RSNA_PATCH_SIZE, in_channels=1),
        ]
