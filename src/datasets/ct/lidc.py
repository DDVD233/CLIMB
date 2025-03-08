import glob
import os
from ast import literal_eval
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pylidc as pl
import torch
import tqdm
from sklearn.model_selection import train_test_split

from src.datasets.specs import Input3dSpec

from .base_ct import CTDatasetBase

# RadFusion
LIDC_DATA_DIR = Path("/home/ubuntu/2022-spr-benchmarking/src/datasets/ct/lidc")
if not LIDC_DATA_DIR.is_dir():
    LIDC_DATA_DIR.mkdir(parents=True, exist_ok=False)
LIDC_DICOM_DIR = LIDC_DATA_DIR / 'LIDC-IDRI'
LIDC_ANNOTATION_DIR = LIDC_DATA_DIR / 'LIDC-XML-only'
LIDC_ANNOTATION_CSV = LIDC_DATA_DIR / 'annotations.csv'

LIDC_STUDY_HDF5 = LIDC_DATA_DIR / "lidc_study.hdf5"
LIDC_DICOM_CSV = LIDC_DATA_DIR / "lidc_2d.csv"
LIDC_WINDOW_CSV = LIDC_DATA_DIR / "lidc_window.csv"

LIDC_PATIENT_COL = "PatientID"
LIDC_STUDY_COL = "StudyInstanceUID"
LIDC_SERIES_COL = "SeriesInstanceUID"
LIDC_INSTANCE_COL = "SOPInstanceUID"
LIDC_INSTANCE_NUM_COL = "SOPInstanceUID"
LIDC_NUM_WINDOW_COL = "NumWindows"
LIDC_NUM_SLICES_COL = "NumSlices"
LIDC_SPLIT_COL = "Split"
LIDC_SMALL_NOD_SLICE_COL = "small_nodule_present_on_image"
LIDC_LARGE_NOD_SLICE_COL = "large_nodule_present_on_image"
LIDC_SMALL_NOD_COL = "small_nodule_present_on_window"
LIDC_LARGE_NOD_COL = "large_nodule_present_on_window"

LIDC_INSTANCE_ORDER_COL = "image_index"
LIDC_LABEL_COL = "label"
LIDC_PATIENT_POSITIION_COL = "ImagePositionPatient_2"

NUM_REVIEWERS = 3
LIDC_WINDOW_IDX_COL = "WindowIdx"
LIDC_WINDOW_SIZE = 24
LIDC_WINDOW_STRIDE = 12
LIDC_MIN_ABNORMAL_SLICE = 4
LIDC_IMAGE_SIZE = 256

LIDC_INPUT_SIZE = (LIDC_IMAGE_SIZE, LIDC_IMAGE_SIZE, LIDC_WINDOW_SIZE)
LIDC_PATCH_SIZE = (16, 16, LIDC_WINDOW_SIZE)
LIDC_IN_CHANNELS = 1
NODULE_WINDOW_CENTER = -600
NODULE_WINDOW_WIDTH = 1500

RANDOM_SEED = 7

from bs4 import BeautifulSoup


class XMLParser:

    def __init__(self, path):
        self.path = path
        with open(path) as f:
            self.soup = BeautifulSoup(f.read(), features="xml")

        self.parsed_dict = self._parse()
        series_uids = self.soup.find_all('SeriesInstanceUid')
        study_uids = self.soup.find_all('StudyInstanceUID')

        if len(series_uids) == 0 or len(study_uids) == 0:
            series_uid = None
            study_uid = None
        else:
            series_uid = self.soup.find_all('SeriesInstanceUid')[0].text
            study_uid = self.soup.find_all('StudyInstanceUID')[0].text
        if len(self.soup.find_all('SeriesInstanceUid')) > 1:
            print("Warning: multiple series in one file")
        if len(self.soup.find_all('StudyInstanceUID')) > 1:
            print("Warning: multiple studyin one file")
        self.parsed_dict['SeriesInstanceUid'] = series_uid
        self.parsed_dict['StudyInstanceUid'] = study_uid

    def pprint(self):
        print(self.soup.prettify())

    def _parse(self):
        self.ret = {}
        readingSessions = self.soup.find_all('readingSession')
        return {
            'readingSession':
                [
                    self._parse_unblindedReadNodule(sess.find_all('unblindedReadNodule'), sess.find_all('nonNodule'))
                    for sess in readingSessions
                ]
        }

    def _parse_unblindedReadNodule(self, unblindedReadNodules, nonNodules):
        nodule_large_than_3mm = [node for node in unblindedReadNodules if node.find('characteristics')]

        nodule_small_than_3mm = [node for node in unblindedReadNodules if not node.find('characteristics')]

        return {
            "large_nodules":
                [
                    {
                        "nodule_feature": self._parse_malignancy(nodule),
                        "roi": [self._parse_roi(roi) for roi in nodule.find_all('roi')]
                    } for nodule in nodule_large_than_3mm
                ],
            "small_nodules":
                [{
                    "roi": [self._parse_roi(roi) for roi in nodule.find_all('roi')]
                } for nodule in nodule_small_than_3mm],
            "unnodles": [{
                "roi": [self._parse_unnodule_roi(unnodule)]
            } for unnodule in nonNodules]
        }

    def _parse_malignancy(self, nodule):
        feature_list = [
            'subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture',
            'malignancy'
        ]
        try:
            feature = [{f: int(nodule.find('characteristics').find(f).string) for f in feature_list}]
            return feature
        except:
            return []

    def _parse_roi(self, roi):
        imageZposition = float(roi.find('imageZposition').string)
        xCoords = [int(x.string) for x in roi.find_all('xCoord')]
        yCoords = [int(x.string) for x in roi.find_all('yCoord')]
        assert len(xCoords) == len(yCoords), self.path
        return {"imageZposition": imageZposition, "coords": list(zip(xCoords, yCoords))}

    def _parse_unnodule_roi(self, unnodule):
        imageZposition = float(unnodule.find('imageZposition').string)
        xCoords = int(unnodule.find('xCoord').string)
        yCoords = int(unnodule.find('yCoord').string)
        #         assert len(xCoords) == len(yCoords), self.path
        return {"imageZposition": imageZposition, "coords": list([xCoords, yCoords])}


def any_exist(files):
    return any(map(os.path.exists, files))


def all_exist(files):
    return all(map(os.path.exists, files))


class LIDCDatasetWindow(CTDatasetBase):
    """A dataset class for the  Lung Image Database Consortium image collection (LIDC-IDRI) 

    This dataset consists of diagnostic and lung cancer screening thoracic 
    computed tomography (CT) scans with marked-up annotated lesions. Seven 
    academic centers and eight medical imaging companies collaborated to create 
    this data set which contains 1018 cases.  Each subject includes images from 
    a clinical thoracic CT scan and an associated XML file that records the 
    results of a two-phase image annotation process performed by four 
    experienced thoracic radiologists. In the initial blinded-read phase, each 
    radiologist independently reviewed each CT scan and marked lesions belonging 
    to one of three categories ("nodule > or =3 mm," "nodule <3 mm," and 
    "non-nodule > or =3 mm").    

    We categorize nodule labels into 3 categories:
        1. Nodule > 3mm: large nodule
        2. Nodule < 3mm: small nodule
        3. Non-nodule: no nodule
    
    While we train the model using windows of slices instead of the full series,
    the final performance is calculated by aggregating prediction probabilities,
    by taking the maximum value, from all windows in a series.

    More information can be found on the website:
        http://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
    """

    def __init__(self, base_root: str = LIDC_DATA_DIR, download: bool = True, train: bool = True):
        super().__init__(train)

        # check if LIDC is downloaded and pre-processed
        self.check_data()

        # literal_eval to convert str to list
        self.df = pd.read_csv(LIDC_WINDOW_CSV)

        if train:
            self.df = self.df[self.df[LIDC_SPLIT_COL] == "train"]
        else:
            self.df = self.df[self.df[LIDC_SPLIT_COL] == "test"]

        self.df['image_index'] = self.df.image_index.apply(lambda x: literal_eval(str(x)))

    def check_data(self):

        if not os.path.exists(LIDC_DICOM_DIR):
            f"""
            Download LIDC DICOM files from the link below by:  
                1) Downloading and installing the NBIA Data Retriever: 
                    https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide
                2) Downloading the TCIA file for LIDC (Data Access -> Data Type Images -> Download): 
                    https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254c9ca370cd5144a44a694e5a77bfd6815
                3) Run the following to download: 
                    /opt/nbia-data-retriever/nbia-data-retriever --cli <location>/<manifest file name>.tcia -d {LIDC_DATA_DIR} -v –f
                4) Update the path to DICOM files in ~/.pylidcrc to {LIDC_DICOM_DIR}
            """

        if not os.path.exists(LIDC_ANNOTATION_CSV):

            if not os.path.exists(LIDC_ANNOTATION_DIR):
                f""" Downloading and unzip the LIDC annotation XML (Data Access 
                -> Data Type Images -> Radiologist Annotations/Segmetations XML 
                format) from the following link and place it at {LIDC_DATA_DIR}: 
                    https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254
                """

            self.preprocess_annotations()

        # if no data is present, prompt the user to download it
        if not all_exist([LIDC_DICOM_CSV, LIDC_STUDY_HDF5]):
            self.preprocess_data()

        # if the data has not been extracted, extract the data, prioritizing the full-res dataset
        if not all_exist([LIDC_WINDOW_CSV]):
            self.process_window_df()

    def __getitem__(self, index):

        row = self.df.iloc[index]
        study_name = row[LIDC_STUDY_COL]
        slice_idx = sorted(row[LIDC_INSTANCE_ORDER_COL])

        # extract and transform window
        window = self.read_from_hdf5(hdf5_path=LIDC_STUDY_HDF5, key=study_name, slice_idx=slice_idx)
        window = self.windowing(window, NODULE_WINDOW_CENTER, NODULE_WINDOW_WIDTH)
        if len(window.shape) == 3:
            window = np.expand_dims(window, 1)

        x = torch.from_numpy(window).float()
        x = x.permute((1, 2, 3, 0))
        x = x.type(torch.FloatTensor)

        # get labels
        y = np.zeros(2)
        if row[LIDC_LARGE_NOD_COL].item() == 1:
            y[0] = 1
        if row[LIDC_SMALL_NOD_COL].item() == 1:
            y[1] = 1
        y = torch.tensor(y).type(torch.IntTensor)

        return index, x, y

    def __len__(self):
        return len(self.df)

    @staticmethod
    def num_classes():
        """Predicting Central Positive and Non-central Positive"""
        return 2

    @staticmethod
    def spec():
        return [
            Input3dSpec(input_size=LIDC_INPUT_SIZE, patch_size=LIDC_PATCH_SIZE, in_channels=1),
        ]

    def process_dicom(self, dcm):
        pixel_array = dcm.pixel_array

        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        pixel_array = pixel_array * slope + intercept

        pixel_array = cv2.resize(pixel_array, (LIDC_IMAGE_SIZE, LIDC_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return pixel_array

    def add_split_to_label_df(
        label_df: pd.DataFrame,
        val_size: float = 0.15,
        test_size: float = 0.15,
        patient_col: str = LIDC_PATIENT_COL,
        split_col: str = LIDC_SPLIT_COL,
    ):
        patients = label_df[patient_col].unique()

        # split between train and val+test
        split_ratio = val_size + test_size
        train_patients, test_val_patients = train_test_split(patients, test_size=split_ratio, random_state=RANDOM_SEED)

        # split between val and test
        test_split_ratio = test_size / (val_size + test_size)
        val_patients, test_patients = train_test_split(test_val_patients, test_size=test_split_ratio, random_state=RANDOM_SEED)
        train_rows = label_df[patient_col].isin(train_patients)
        label_df.loc[train_rows, split_col] = "train"
        val_rows = label_df[patient_col].isin(val_patients)
        label_df.loc[val_rows, split_col] = "valid"
        test_rows = label_df[patient_col].isin(test_patients)
        label_df.loc[test_rows, split_col] = "test"

        return label_df

    def preprocess_data(self):
        records = []

        annotation_df = pd.read_csv(LIDC_ANNOTATION_CSV)

        nodule_dict = annotation_df[annotation_df["parse_type"] == "nodule"].groupby(["seriesuid", "imageZposition"]
                                                                                    )['doctor_id'].apply(set).to_dict()
        large_nodule_dict = annotation_df[annotation_df["parse_type"] == "large_nodule"].groupby(
            ["seriesuid", "imageZposition"]
        )['doctor_id'].apply(set).to_dict()

        for scan in tqdm.tqdm(pl.query(pl.Scan).all()):
            dicoms = scan.load_all_dicom_images(verbose=False)
            if len(dicoms) == 0:
                continue

            for i, dcm in enumerate(dicoms):

                small_nodule, large_nodule = 0, 0
                instance_id = (dcm.SeriesInstanceUID, dcm.ImagePositionPatient[2])
                if instance_id in nodule_dict:
                    if len(nodule_dict[instance_id]) >= NUM_REVIEWERS:
                        small_nodule = 1
                if instance_id in large_nodule_dict:
                    if len(large_nodule_dict[instance_id]) >= NUM_REVIEWERS:
                        large_nodule = 1

                records.append(
                    {
                        LIDC_PATIENT_COL: dcm.PatientID,
                        LIDC_STUDY_COL: dcm.StudyInstanceUID,
                        LIDC_SERIES_COL: dcm.SeriesInstanceUID,
                        LIDC_INSTANCE_NUM_COL: dcm.SOPInstanceUID,
                        LIDC_INSTANCE_ORDER_COL: i,
                        LIDC_SMALL_NOD_SLICE_COL: small_nodule,
                        LIDC_LARGE_NOD_SLICE_COL: large_nodule,
                        LIDC_INSTANCE_NUM_COL: dcm.InstanceNumber,
                        LIDC_PATIENT_POSITIION_COL: dcm.ImagePositionPatient[2],
                        LIDC_NUM_SLICES_COL: len(dicoms)
                    }
                )

        df = pd.DataFrame.from_records(records)
        df = self.add_split_to_label_df(df)
        df.to_csv(LIDC_DICOM_CSV, index=False)

    def process_window_df(num_slices: int = 24, min_abnormal_slice: int = 4):

        # read in df
        df = pd.read_csv(LIDC_DICOM_CSV)

        # count number of windows per slice
        count_num_windows = lambda x: x // num_slices\
            + (1 if x % num_slices > 0 else 0)
        df[LIDC_NUM_WINDOW_COL] = df[LIDC_NUM_SLICES_COL].apply(count_num_windows)

        # get windows list
        df_study = df.groupby([LIDC_STUDY_COL]).head(1)
        window_labels = defaultdict(list)

        # studies
        for _, row in tqdm.tqdm(df_study.iterrows(), total=df_study.shape[0]):
            study_name = row[LIDC_STUDY_COL]
            split = row[LIDC_SPLIT_COL]
            study_df = df[df[LIDC_STUDY_COL] == study_name]
            study_df = study_df.sort_values(LIDC_PATIENT_POSITIION_COL)

            # windows
            for idx in range(row[LIDC_NUM_WINDOW_COL]):
                start_idx = idx * num_slices
                end_idx = (idx + 1) * num_slices

                window_df = study_df.iloc[start_idx:end_idx]
                num_small_nod = window_df[LIDC_SMALL_NOD_SLICE_COL].sum()
                num_large_nod = window_df[LIDC_LARGE_NOD_SLICE_COL].sum()
                small_label = 1 if num_small_nod >= min_abnormal_slice else 0
                large_label = 1 if num_large_nod >= min_abnormal_slice else 0
                window_labels[LIDC_SMALL_NOD_COL].append(small_label)
                window_labels[LIDC_LARGE_NOD_COL].append(large_label)
                window_labels[LIDC_STUDY_COL].append(study_name)
                window_labels[LIDC_WINDOW_IDX_COL].append(idx)
                window_labels[LIDC_SPLIT_COL].append(split)
                window_labels[LIDC_PATIENT_POSITIION_COL].append(window_df[LIDC_PATIENT_POSITIION_COL].tolist())
                window_labels[LIDC_SMALL_NOD_SLICE_COL].append(window_df[LIDC_SMALL_NOD_SLICE_COL].tolist())
                window_labels[LIDC_LARGE_NOD_SLICE_COL].append(window_df[LIDC_LARGE_NOD_SLICE_COL].tolist())
                window_labels[LIDC_INSTANCE_COL].append(window_df[LIDC_INSTANCE_COL].tolist())
                window_labels[LIDC_INSTANCE_ORDER_COL].append(window_df[LIDC_INSTANCE_ORDER_COL].tolist())

        df_window = pd.DataFrame.from_dict(window_labels)
        df_window.to_csv(LIDC_WINDOW_CSV)

    def preprocess_annotations(self):

        all_annotations = []

        # get all XML files
        xml_paths = glob.glob(str(LIDC_ANNOTATION_DIR / "tcia-lidc-xml/*/*.xml"))

        for p in tqdm.tqdm(xml_paths, total=len(xml_paths)):
            xmlparse = XMLParser(p)
            json_data = xmlparse.parsed_dict
            study_uid = json_data['StudyInstanceUid']
            series_uid = json_data['SeriesInstanceUid']

            if series_uid is None or study_uid is None:
                continue

            df_unnodule_list = []
            df_nodule_list = []
            df_large_nodule_list = []

            for index, readingSession in enumerate(json_data['readingSession']):
                for unnodule in readingSession["unnodles"]:
                    for roi in unnodule['roi']:
                        df_unnodule_list.append([index, roi['imageZposition'], roi['coords'][1], roi['coords'][0]])

            for index, readingSession in enumerate(json_data['readingSession']):
                for nodule in readingSession["small_nodules"]:
                    for roi in nodule['roi']:
                        df_nodule_list.append([index, roi['imageZposition'], roi['coords'][0][1], roi['coords'][0][0]])

            for index, readingSession in enumerate(json_data['readingSession']):
                for nodule in readingSession["large_nodules"]:
                    for roi in nodule['roi']:
                        df_large_nodule_list.append([index, roi['imageZposition'], roi['coords'][0][1], roi['coords'][0][0]])

            df_unnodule = pd.DataFrame(df_unnodule_list, columns=['doctor_id', "imageZposition", 'coordy', 'coordx'])
            df_unnodule['parse_type'] = "unnodule"
            df_nodule = pd.DataFrame(df_nodule_list, columns=['doctor_id', "imageZposition", 'coordy', 'coordx'])
            df_nodule['parse_type'] = "nodule"
            df_large_nodule = pd.DataFrame(df_large_nodule_list, columns=(['doctor_id', "imageZposition", 'coordy', 'coordx']))
            df_large_nodule['parse_type'] = "large_nodule"
            nodule_info = pd.concat([df_unnodule, df_nodule, df_large_nodule]).reset_index(drop=True)
            nodule_info['seriesuid'] = series_uid
            nodule_info['studyuid'] = study_uid
            newcolumns = ['seriesuid', 'studyuid', 'doctor_id', 'parse_type', "imageZposition", 'coordy', 'coordx']
            all_annotations.append(nodule_info[newcolumns])

        df = pd.concat(all_annotations).reset_index(drop=True)
        df.to_csv(LIDC_ANNOTATION_CSV)
