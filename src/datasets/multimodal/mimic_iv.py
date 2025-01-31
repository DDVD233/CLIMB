import copy
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import ujson
import wfdb
from scipy.signal import resample
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.io as sio
import traceback


class MIMIC_IV(Dataset):
    '''
    A dataset class for the MIMIC-IV dataset integrated with MIMIC-CXR images.
    Maintains original data values (numbers and strings) for interpretability.
    '''

    def __init__(self,
                 base_root: str,
                 split: str = 'train',
                 included_tables: Optional[List[str]] = None,
                 save_individual_files: bool = True):
        """
        Initialize MIMIC-IV dataset.

        Args:
            root: Base root directory
            split: Data split ('train', 'validate', or 'test')
            included_tables: List of tables to include in the dataset
            save_individual_files: Whether to save individual patient JSON files
        """
        self.root = base_root
        self.mimiciv_root = os.path.join(base_root, 'multimodal', 'mimiciv')
        self.cxr_root = os.path.join(base_root, 'chest_xray', 'mimic-cxr')
        self.ecg_root = os.path.join(base_root, 'ecg', 'mimiciv')
        self.split = split
        self.save_individual_files = save_individual_files

        # Create data directory for individual patient files
        if save_individual_files:
            self.data_dir = os.path.join(self.mimiciv_root, 'data')
            self.admissions_dir = os.path.join(self.mimiciv_root, 'admissions_multimodal')
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.admissions_dir, exist_ok=True)

        # Default tables to include
        self.included_tables = included_tables or [
            'patients', 'admissions', 'labevents', 'chartevents',
            'diagnoses_icd', 'procedures_icd', 'prescriptions'
        ]

        self.icd_code_to_description = {}
        if os.path.exists(os.path.join(self.mimiciv_root, 'icd_code_to_description.json')):
            with open(os.path.join(self.mimiciv_root, 'icd_code_to_description.json'), 'r') as f:
                self.icd_code_to_description = ujson.load(f)

        self._load_ecg_metadata()

        # Build the dataset index
        # self.patient_data = {}
        self.subject_ids = []
        self.build_index()

    def _load_ecg_metadata(self):
        """Load ECG-related metadata."""
        # Load machine measurements
        self.ecg_measurements = pd.read_csv(
            os.path.join(self.ecg_root, 'machine_measurements.csv'), low_memory=False
        )

        # Load waveform-note links
        self.ecg_notes = pd.read_csv(
            os.path.join(self.ecg_root, 'waveform_note_links.csv'), low_memory=False
        )

        # Load record list
        self.ecg_records = pd.read_csv(
            os.path.join(self.ecg_root, 'record_list.csv'), low_memory=False
        )

    def _get_ecg_data(self, subject_id: int) -> List[Dict]:
        """Get all ECG data for a patient, resample to 500Hz, and save as new file."""
        ecg_data = []
    
        # Get all records for this patient
        patient_records = self.ecg_records[self.ecg_records['subject_id'] == subject_id]
    
        for _, record in patient_records.iterrows():
            study_id = record['study_id']
            record_path = os.path.join(self.ecg_root, record['path'].replace('.hea', ''))

            os.makedirs(os.path.join(self.mimiciv_root, 'resampled'), exist_ok=True)
            resampled_file = os.path.join(self.mimiciv_root, 'resampled', f"{subject_id}_{study_id}.mat")
            if os.path.exists(resampled_file):
                relative_path = os.path.relpath(resampled_file, self.mimiciv_root)
                ecg_data.append({
                    'file_path': relative_path,
                    'study_id': str(study_id),
                    'sampling_rate': 500,
                    'report_id': None
                })
                continue
    
            try:
                # Read ECG record
                record_data = wfdb.rdrecord(record_path)
    
                # Resample signals to 500 Hz
                target_fs = 500
                if record_data.fs != target_fs:
                    resampled_signals = resample(record_data.p_signal, int(len(record_data.p_signal) * target_fs / record_data.fs), axis=0)
                else:
                    resampled_signals = record_data.p_signal
                # 5000, 12 -> 12, 5000
                resampled_signals = resampled_signals.T
    
                # Get machine measurements
                measurements = self.ecg_measurements[
                    (self.ecg_measurements['subject_id'] == subject_id) &
                    (self.ecg_measurements['study_id'] == study_id)
                    ].to_dict('records')
    
                # Get report link if available
                report = self.ecg_notes[
                    (self.ecg_notes['subject_id'] == subject_id) &
                    (self.ecg_notes['study_id'] == study_id)
                    ].to_dict('records')
    
                # Save resampled ECG data as a new mat file, don't use wfdb
                sio.savemat(resampled_file, {
                    'val': resampled_signals,
                    'fs': target_fs,
                    'measurements': measurements
                })

                relative_path = os.path.relpath(resampled_file, self.mimiciv_root)
    
                ecg_data.append({
                    'file_path': relative_path,
                    'study_id': str(study_id),
                    'timestamp': record_data.base_datetime.isoformat() if record_data.base_datetime else None,
                    'sampling_rate': target_fs,
                    'report_id': report[0]['note_id'] if report else None
                })
    
            except Exception as e:
                print(f"Error loading ECG {record_path}: {e}")
                # print traceback
                traceback.print_exc()
                continue
    
        return ecg_data

    def _load_metadata(self):
        """Load metadata tables for interpretable values."""
        # Load D_ICD_DIAGNOSES for diagnosis descriptions
        try:
            d_icd_diagnoses = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/d_icd_diagnoses.csv.gz'))
            self.diagnosis_descriptions = dict(zip(d_icd_diagnoses['icd_code'], d_icd_diagnoses['long_title']))
        except:
            self.diagnosis_descriptions = {}
            print("Warning: Could not load diagnosis descriptions")

        # Load D_ICD_PROCEDURES for procedure descriptions
        try:
            d_icd_procedures = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/d_icd_procedures.csv.gz'))
            self.procedure_descriptions = dict(zip(d_icd_procedures['icd_code'], d_icd_procedures['long_title']))
        except:
            self.procedure_descriptions = {}
            print("Warning: Could not load procedure descriptions")

        # Load D_LABITEMS for lab test descriptions
        try:
            d_labitems = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/d_labitems.csv.gz'))
            self.lab_descriptions = dict(zip(d_labitems['itemid'], d_labitems['label']))
        except:
            self.lab_descriptions = {}
            print("Warning: Could not load lab descriptions")

        # Load D_ITEMS for chart event descriptions
        try:
            d_items = pd.read_csv(os.path.join(self.mimiciv_root, 'icu/d_items.csv.gz'))
            self.item_descriptions = dict(zip(d_items['itemid'], d_items['label']))
        except:
            self.item_descriptions = {}
            print("Warning: Could not load item descriptions")

    def build_index(self):
        """Build patient dictionary with all available data."""
        print("Building MIMIC-IV index...")

        # Skip if already built
        datas = os.listdir(self.data_dir)
        admissions = os.listdir(self.admissions_dir)
        if len(datas) > 0 and len(admissions) == 0:
            print(f"Dataset already built with {len(datas)} patients")
            self.subject_ids = [int(data.split('.')[0]) for data in datas]
            total_count, multimodal_count = 0, 0
            for data in tqdm(datas, desc="Loading patient data"):
                with open(os.path.join(self.data_dir, data), 'r') as f:
                    patient_info = ujson.load(f)
                admissions = patient_info['admissions']

                # attach ecg data (which was added later)
                subject_id = int(data.split('.')[0])
                ecg_data = self._get_ecg_data(subject_id)
                patient_info['ecg_data'] = ecg_data

                if len(ecg_data) == 0:
                    continue

                for admission in admissions:
                    hadm_id = int(admission['hadm_id'])
                    admit_time = datetime.strptime(admission['admittime'], '%Y-%m-%d %H:%M:%S')
                    # visible admissions are those that are not after this admit time
                    visible_ham_ids = [hadm_id]
                    # for admission2 in admissions:
                    #     if datetime.strptime(admission2['admittime'], '%Y-%m-%d %H:%M:%S') < admit_time:
                    #         visible_ham_ids.append(admission2['hadm_id'])
                    input_data: dict = copy.deepcopy(patient_info)
                    # drop everything that is not visible
                    diagnoses = input_data.pop('diagnoses')
                    outputs = []
                    for diagnosis in diagnoses:
                        if diagnosis['hadm_id'] == hadm_id:  # only keep the diagnoses for this admission
                            if diagnosis['icd_code'] in self.icd_code_to_description:
                                outputs.append(self.icd_code_to_description[diagnosis['icd_code']]['index'])
                                self.icd_code_to_description[diagnosis['icd_code']]['count'] += 1
                            else:
                                mapping = {
                                    'index': len(self.icd_code_to_description),
                                    'description': diagnosis['description'],
                                    'count': 1
                                }
                                self.icd_code_to_description[diagnosis['icd_code']] = mapping

                                with open(os.path.join(self.mimiciv_root, 'icd_code_to_description.json'), 'w') as f:
                                    ujson.dump(self.icd_code_to_description, f)
                    if len(outputs) == 0:
                        continue
                    lab_values = input_data['lab_values']
                    self.remove_invisibles(lab_values, visible_ham_ids)

                    chart_values = input_data['chart_values']
                    self.remove_invisibles(chart_values, visible_ham_ids)

                    prescriptions = input_data['prescriptions']
                    prescriptions = [x for x in prescriptions if x['hadm_id'] in visible_ham_ids]
                    input_data['prescriptions'] = prescriptions
                    prescription_lists = [x['drug'] for x in prescriptions if x['hadm_id'] in visible_ham_ids]
                    prescription_lists = list(set(prescription_lists))

                    admissions = input_data.pop('admissions')
                    for admission in admissions:
                        if admission['hadm_id'] != hadm_id:
                            continue
                        input_data['admissions'] = [admission]

                    # if (len(input_data['image_paths']) == 0 and len(input_data['ecg_data']) == 0 and
                    #         len(input_data['lab_values'].keys()) == 0 and len(input_data['chart_values']) == 0):
                    #     # No useful information
                    #     continue
                    total_count += 1
                    if len(input_data['image_paths']) == 0 or len(input_data['ecg_data']) == 0:
                        # Not multimodal useful information
                        continue
                    multimodal_count += 1
                    survived = 1
                    if admission['deathtime'] is not None:
                        survived = 0
                    ihm_48 = 1
                    if survived == 0 and 'deathtime' in admission:
                        death_time = datetime.strptime(admission['deathtime'], '%Y-%m-%d %H:%M:%S')
                        if death_time - admit_time < timedelta(days=2):
                            ihm_48 = 0
                    image_paths = input_data.pop('image_paths')
                    ecg_data = input_data.pop('ecg_data')
                    input_data['prescription_short'] = prescription_lists
                    admission_data = {
                        'input': input_data,
                        'outputs': outputs,
                        'survived': survived,
                        '48_ihm': ihm_48,
                        'image_paths': image_paths,
                        'ecg_data': ecg_data
                    }

                    with open(os.path.join(self.admissions_dir, f"{hadm_id}.json"), 'w') as f:
                        ujson.dump(admission_data, f)
            print(f"Total count: {total_count}, multimodal count: {multimodal_count}")
            with open(os.path.join(self.mimiciv_root, 'icd_code_to_description.json'), 'w') as f:
                ujson.dump(self.icd_code_to_description, f)
            return
        elif len(datas) > 0 and len(admissions) > 0:
            print(f"Dataset already built with {len(datas)} patients")
            admissions = [int(admission.split('.')[0]) for admission in admissions]
            admissions = sorted(admissions)
            if self.split == 'train':
                self.subject_ids = admissions[:int(0.8 * len(admissions))]
            else:
                self.subject_ids = admissions[int(0.8 * len(admissions)):]
            print(f"Split {self.split} with {len(self.subject_ids)} patients. "
                  f"First 10 ids are {self.subject_ids[:10]}")
        else:
            self._load_metadata()

            # Initialize patient dictionary with nested defaultdict
            patient_dict = defaultdict(lambda: {
                'demographics': {},
                'admissions': [],
                'diagnoses': [],
                'procedures': [],
                'lab_values': defaultdict(list),
                'chart_values': defaultdict(list),
                'prescriptions': [],
                'image_paths': [],
                'ecg_data': []
            })

            # Link to MIMIC-CXR
            cxr_splits = pd.read_csv(os.path.join(self.cxr_root, "mimic-cxr-2.0.0-split.csv.gz"))
            cxr_study_list = pd.read_csv(os.path.join(self.cxr_root, "cxr-study-list.csv.gz"))

            # Merge CXR information
            cxr_data = pd.merge(cxr_study_list, cxr_splits, on=['subject_id', 'study_id'])
            for _, row in tqdm(cxr_data.iterrows(), total=len(cxr_data), desc="Linking to CXR"):
                img_path = f"files/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg"
                patient_dict[row['subject_id']]['image_paths'].append({
                    'path': img_path,
                    'study_id': str(row['study_id']),
                    'dicom_id': str(row['dicom_id'])
                })

            # print("Adding ECG data...")
            for subject_id in tqdm(patient_dict.keys()):
                ecg_data = self._get_ecg_data(subject_id)
                patient_dict[subject_id]['ecg_data'] = ecg_data

            # Load and process core patient data
            patients_df = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/patients.csv.gz'))
            for _, row in tqdm(patients_df.iterrows(), total=len(patients_df), desc="Processing patients"):
                patient_dict[row['subject_id']]['demographics'] = {
                    'gender': row['gender'],
                    'anchor_age': float(row['anchor_age']),
                    'anchor_year': int(row['anchor_year']),
                    'anchor_year_group': row['anchor_year_group']
                }

            # Add admission information
            if 'admissions' in self.included_tables:
                admissions_df = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/admissions.csv.gz'))
                for _, row in tqdm(admissions_df.iterrows(), total=len(admissions_df), desc="Processing admissions"):
                    admission_data = {
                        'hadm_id': int(row['hadm_id']),
                        'admittime': row['admittime'],
                        'dischtime': row['dischtime'],
                        'deathtime': row['deathtime'] if pd.notna(row['deathtime']) else None,
                        'admission_type': row['admission_type'],
                        'admission_location': row['admission_location'],
                        'discharge_location': row['discharge_location'],
                        'insurance': row['insurance']
                    }
                    patient_dict[row['subject_id']]['admissions'].append(admission_data)

            # Add diagnoses with descriptions
            if 'diagnoses_icd' in self.included_tables:
                diagnoses_df = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/diagnoses_icd.csv.gz'))
                for _, row in tqdm(diagnoses_df.iterrows(), total=len(diagnoses_df), desc="Processing diagnoses"):
                    diagnosis = {
                        'icd_code': row['icd_code'],
                        'icd_version': row['icd_version'],
                        'description': self.diagnosis_descriptions.get(row['icd_code'], ''),
                        'seq_num': int(row['seq_num']),
                        'hadm_id': int(row['hadm_id'])
                    }
                    patient_dict[row['subject_id']]['diagnoses'].append(diagnosis)

            # Add procedures with descriptions
            if 'procedures_icd' in self.included_tables:
                procedures_df = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/procedures_icd.csv.gz'))
                for _, row in tqdm(procedures_df.iterrows(), total=len(procedures_df), desc="Processing procedures"):
                    procedure = {
                        'icd_code': row['icd_code'],
                        'icd_version': row['icd_version'],
                        'description': self.procedure_descriptions.get(row['icd_code'], ''),
                        'seq_num': int(row['seq_num']),
                        'hadm_id': int(row['hadm_id'])
                    }
                    patient_dict[row['subject_id']]['procedures'].append(procedure)

            # Add lab events with descriptions
            if 'labevents' in self.included_tables:
                labevents_df = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/labevents.csv.gz'))
                for _, row in tqdm(labevents_df.iterrows(), total=len(labevents_df), desc="Processing lab events"):
                    if pd.notna(row['valuenum']):
                        lab_name = self.lab_descriptions.get(row['itemid'], str(row['itemid']))
                        patient_dict[row['subject_id']]['lab_values'][lab_name].append({
                            'time': row['charttime'],
                            'value': float(row['valuenum']),
                            'unit': row['valueuom'] if pd.notna(row['valueuom']) else None,
                            'hadm_id': int(row['hadm_id']) if pd.notna(row['hadm_id']) else None
                        })

            # Add chart events with descriptions
            if 'chartevents' in self.included_tables:
                chartevents_df = pd.read_csv(os.path.join(self.mimiciv_root, 'icu/chartevents.csv.gz'))
                for _, row in tqdm(chartevents_df.iterrows(), total=len(chartevents_df), desc="Processing chart events"):
                    if pd.notna(row['valuenum']):
                        chart_name = self.item_descriptions.get(row['itemid'], str(row['itemid']))
                        patient_dict[row['subject_id']]['chart_values'][chart_name].append({
                            'time': row['charttime'],
                            'value': float(row['valuenum']),
                            'unit': row['valueuom'] if pd.notna(row['valueuom']) else None,
                            'hadm_id': int(row['hadm_id']) if pd.notna(row['hadm_id']) else None
                        })

            # Add prescriptions
            if 'prescriptions' in self.included_tables:
                prescriptions_df = pd.read_csv(os.path.join(self.mimiciv_root, 'hosp/prescriptions.csv.gz'))
                for _, row in tqdm(prescriptions_df.iterrows(), total=len(prescriptions_df), desc="Processing prescriptions"):
                    prescription = {
                        'drug': row['drug'],
                        'drug_type': row['drug_type'],
                        'route': row['route'],
                        'hadm_id': int(row['hadm_id']),
                        'starttime': row['starttime'],
                        'stoptime': row['stoptime']
                    }
                    patient_dict[row['subject_id']]['prescriptions'].append(prescription)

            # Convert defaultdict to regular dict
            self.patient_data = dict(patient_dict)
            self.subject_ids = sorted(self.patient_data.keys())

            if self.save_individual_files:
                print("Saving individual patient files...")
                for subject_id in tqdm(self.subject_ids):
                    file_path = os.path.join(self.data_dir, f"{subject_id}.json")
                    with open(file_path, 'w') as f:
                        ujson.dump(self.patient_data[subject_id], f)

            print(f"Dataset built with {len(self.subject_ids)} patients")

    def append_ecg_data(self):
        """Append ECG data to patient dictionary."""

        # Load ECG metadata
        self._load_ecg_metadata()

        # Add ECG data to patient dictionary
        for subject_id in tqdm(self.subject_ids, desc="Adding ECG data"):
            ecg_data = self._get_ecg_data(subject_id)
            self.patient_data[subject_id]['ecg_data'] = ecg_data

            if self.save_individual_files:
                file_path = os.path.join(self.data_dir, f"{subject_id}.json")
                with open(file_path, 'w') as f:
                    ujson.dump(self.patient_data[subject_id], f)

    def remove_invisibles(self, lab_values, visible_ham_ids):
        for lab_name, lab_data in lab_values.items():
            new_lab_data = []
            for lab in lab_data:
                if lab['hadm_id'] in visible_ham_ids:
                    new_lab_data.append(lab)
            lab_values[lab_name] = new_lab_data

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __num_classes__(self) -> int:
        return len(self.icd_code_to_description)

    def __getitem__(self, index: int) -> Dict:
        """
        Get a patient's data with all original values maintained.

        Returns:
            Dictionary containing all patient information and image paths
        """
        subject_id = self.subject_ids[index]
        # patient_data = self.patient_data[subject_id]
        with open(os.path.join(self.admissions_dir, f"{subject_id}.json"), 'r') as f:
            patient_data = ujson.load(f)

        # Add full image paths
        image_paths = [
            {**img_data, 'full_path': os.path.join(self.cxr_root, img_data['path'])}
            for img_data in patient_data['image_paths']
        ]
        patient_data['image_paths'] = image_paths
        patient_data['diagnoses'] = patient_data.pop('outputs')

        admissions = patient_data['input']['admissions']
        latest_admission = sorted(admissions, key=lambda x: datetime.strptime(x['admittime'], '%Y-%m-%d %H:%M:%S'))[-1]
        length_of_stay = datetime.strptime(latest_admission['dischtime'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(latest_admission['admittime'], '%Y-%m-%d %H:%M:%S')

        # Exact length of stay in days, could be non-integer
        patient_data['length_of_stay'] = length_of_stay.total_seconds() / (24 * 60 * 60)

        return patient_data


if __name__ == '__main__':
    base_dir = '/mnt/8T/high_modality/'

    mimic_iv = MIMIC_IV(base_root=base_dir, split='train')
    data = mimic_iv[0]
    mimic_iv_val = MIMIC_IV(base_root=base_dir, split='valid')