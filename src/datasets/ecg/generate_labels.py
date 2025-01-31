import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

"""
A python file for generating labels for the ECG data.
"""

def load_header(header_file):
    """
    Returns the header of a PTB-XL recording.
    """
    with open(header_file, 'r') as f:
        header = f.read()
    return header


def get_labels(header):
    """
    Returns a list of labels from the header of a PTB-XL recording.
    """
    labels = list()
    for l in header.split('\n'):
        if l.startswith('#Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    labels.append(entry.strip())
            except:
                pass
    return labels


def find_challenge_files(data_directory):
    """
    Returns a list of header and recording files in the PTB-XL dataset.
    """
    header_files = list()
    recording_files = list()
    for f in sorted(os.listdir(data_directory)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension == '.hea':
            header_file = os.path.join(data_directory, root + '.hea')
            recording_file = os.path.join(data_directory, root + '.mat')
            if os.path.isfile(header_file) and os.path.isfile(recording_file):
                header_files.append(header_file)
                recording_files.append(recording_file)
    return header_files, recording_files


def find_header_files(data_directory):
    """
    Returns a list of header files in the PTB-XL dataset.
    """
    header_files = list()
    for f in sorted(os.listdir(data_directory)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension == '.hea':
            header_file = os.path.join(data_directory, root + '.hea')
            if os.path.isfile(header_file):
                header_files.append(header_file)
    return header_files


def get_distribution(directory):
    """
    Returns a dictionary with the distribution of labels in the PTB-XL dataset.
    """
    header_files = find_header_files(directory)
    set_of_labels = list()
    for h in header_files:
        header = load_header(h)
        for l in header.split('\n'):
            if l.startswith('#Dx'):
                try:
                    entries = l.split(': ')[1].split(',')
                    for entry in entries:
                        set_of_labels.append(entry.strip())
                except:
                    pass
    print("#recordings: ", len(header_files))
    return Counter(set_of_labels)


###Generate label and splits for PTBXL###
ptbxl = {
    233917008: '_AVB',
    270492004: 'AVB',
    195042002: '2AVB',
    27885002: '3AVB',
    164951009: 'HVOLT',
    164889003: 'AFIB',
    164890007: 'AFLT',
    39732003: 'LAD',
    164865005: 'PMI',
    54329005: 'AMI',
    428750005: 'STTC',
    47665007: 'RAD',
    251200008: 'SAG',
    11157007: 'BIGU',
    65778007: 'CD',
    164909002: 'CLBBB',
    713427006: 'CRBBB',
    266249003: 'SEHYP',
    251120003: 'ILBBB',
    59931005: 'INVT',
    713426002: 'IRBBB',
    164861001: 'ISCIL',
    426434006: 'ISCAN',
    425419005: 'ISCIN',
    425623009: 'ISCLA',
    698252002: 'IVCD',
    445118002: 'LAFB/LPFB',
    67741000119109: 'LAO/LAE',
    111975006: 'LNGQT',
    164934002: 'TAB_',
    445211001: 'LPFB',
    164947007: 'LPR',
    164873001: 'VCLVH',
    251146004: 'LVOLT',
    426783006: 'SR',
    55930002: 'NST_',
    284470004: 'PAC',
    10370003: 'PACE',
    164884008: 'PVC',
    67198005: 'PSVT',
    164917005: 'QWAVE',
    446358003: 'RAO/RAE',
    89792004: 'RVH',
    427393009: 'SARRH',
    426177001: 'SBRAD',
    427084000: 'STACH',
    429622005: 'STD_',
    164931005: 'STE_',
    63593006: 'SVARR',
    426761007: 'SVTAC',
    251180001: 'TRIGU',
    74390002: 'WPW'
}
normal = ['NORM', 'SARRH', 'SBRAD', 'SR', 'STACH']
cd = [
    'AVB', '1AVB', '2AVB', '3AVB', 'CD', 'CLBBB', 'CRBBB', 'ILBBB', 'IRBBB',
    'IVCB', 'IVCD', 'LAFB', 'LAFB/LPFB', 'LPFB', 'LPR', 'PSVT', 'SVARR',
    'SVTAC', 'WPW'
]
hyp = [
    'HYP', 'ALAD', 'LAD', 'LAO/LAE', 'LVH', 'RAD', 'RHV', 'RVH', 'RAO/RAE',
    'SEHYP', 'VCLVH'
]

mi = [
    'AMI', 'ALMI', 'ASMI', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJLA',
    'INVT', 'IPLMI', 'IPMI', 'LMI', 'MI', 'PMI'
]
sttc = [
    'ANEUR', 'DIG', 'EL', 'ISC_', 'ISCA', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCI',
    'ISCIL', 'ISCIN', 'ISCLA', 'LNGQT', 'NDT', 'NST_', 'NT_', 'STD_', 'STE_',
    'STTC', 'TAB_'
]
afib = ['AFIB', 'AFLT']

other = [
    'ABQRS', 'ARAD', 'AXL', 'AXR\t', 'BIGU', 'HVOLT', 'LOWT', 'LVOLT', 'PACE',
    'PAC', 'PRC(S)', 'PVC', 'QWAVE', 'SAG', 'TRIGU'
]

map1 = {
    'normal': normal,
    'cd': cd,
    'mi': mi,
    'sttc': sttc,
    'other': other,
    'afib': afib,
    'hyp': hyp
}

def get_labels(directory):
    """
    Returns a dataframe with the labels for each recording
    """
    header_files = find_header_files(directory)
    df = pd.DataFrame(columns=[
        'file_name', 'normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp'
    ])
    for i in tqdm(range(len(header_files))):
        h = header_files[i]
        set_of_labels = list()
        header = load_header(h)
        for l in header.split('\n'):
            if l.startswith('#Dx'):
                try:
                    entries = l.split(': ')[1].split(',')
                    for entry in entries:
                        set_of_labels.append(entry.strip())
                except:
                    pass
        row = pd.DataFrame(columns=[
            'file_name', 'normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp'
        ])

        row.loc[0, 'file_name'] = h

        for c in ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']:
            row[c] = 0
            for s in set_of_labels:
                s = int(s)
                if ptbxl[s] in map1[c]:
                    row[c] = 1
        df = pd.concat([df, row]).reset_index(drop=True)

    return df


# Generate labels for the PTB-XL dataset
a = get_labels('/home/ubuntu/raw/ecg/ptbxl')
a['patient'] = a.file_name.str.upper().apply(
    lambda x: x.split('/')[-1].split('.')[0])
a = pd.read_csv("ptbxl.csv")
a = pd.read_csv("ptbxl.csv")
TRAIN_SPLIT_RATIO = 0.8
print(a.head(10))
unique_counts = a[['normal', 'cd', 'mi', 'sttc', 'other', 'afib',
                   'hyp']].value_counts()
index = pd.DataFrame(columns=a.columns.values)
for c_value, l in unique_counts.items():
    s = ((a['normal'] == c_value[0]) & (a['cd'] == c_value[1]) &
         (a['mi'] == c_value[2]) & (a['sttc'] == c_value[3]) &
         (a['other'] == c_value[4]) & (a['afib'] == c_value[5]) &
         (a['hyp'] == c_value[6]))
    df_sub = a[s]
    g = df_sub.sample(frac=TRAIN_SPLIT_RATIO, replace=False)
    index = pd.concat([index, g]).reset_index(drop=True)
train_patients = index['patient'].values.tolist()
a['split'] = 'val'
for i in range(a.shape[0]):
    if a.loc[i, 'patient'] in train_patients:
        a.loc[i, 'split'] = 'train'
a['label'] = -1
for i in tqdm(range(a.shape[0])):
    np.random.seed(i)
    b = a.loc[i, ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']].values
    a.loc[i, 'label'] = np.random.choice(np.flatnonzero(b == b.max()))
print(a.head(10))
print(a.label.value_counts())

a.to_csv("ptbxl_splits.csv")

### Generate labels for the Georgia dataset###
ga = {
    270492004: '1st degree av block',
    195042002: '2nd degree av block',
    164951009: 'abnormal QRS',
    61277005: 'accelerated idioventricular rhythm',
    426664006: 'accelerated junctional rhythm',
    57054005: 'acute myocardial infarction',
    413444003: 'acute myocardial ischemia',
    426434006: 'anterior ischemia',
    54329005: 'anterior myocardial infarction',
    251173003: 'atrial bigeminy',
    164889003: 'atrial fibrillation',
    195080001: 'atrial fibrillation and flutter',
    164890007: 'atrial flutter',
    195126007: 'atrial hypertrophy',
    251268003: 'atrial pacing pattern',
    713422000: 'atrial tachycardia',
    233917008: 'av block',
    251170000: 'blocked premature atrial contraction\xa0',
    74615001: 'brady tachy syndrome',
    426627000: 'bradycardia',
    418818005: 'brugada syndrome',
    6374002: 'bundle branch block',
    698247007: 'cardiac dysrhythmia',
    426749004: 'chronic atrial fibrillation',
    413844008: 'chronic myocardial ischemia',
    78069008: 'chronic rheumatic pericarditis',
    27885002: 'complete heart block',
    713427006: 'complete right bundle branch block',
    204384007: 'congenital incomplete atrioventricular heart block',
    53741008: 'coronary heart disease',
    77867006: 'decreased qt interval',
    82226007: 'diffuse intraventricular block',
    428417006: 'early repolarization',
    251143007: 'ecg artefacts',
    29320008: 'ectopic rhythm',
    423863005: 'electrical alternans',
    13640000: 'fusion beats',
    84114007: 'heart failure',
    368009: 'heart valve disorder',
    251259000: 'high t voltage',
    49260003: 'idioventricular rhythm',
    251120003: 'incomplete left bundle branch block',
    713426002: 'incomplete right bundle branch block',
    251200008: 'indeterminate cardiac axis',
    425419005: 'inferior ischaemia',
    704997005: 'inferior st segment depression',
    50799005: 'isorhythmic dissociation',
    426995002: 'junctional escape',
    251164006: 'junctional premature complex',
    426648003: 'junctional tachycardia',
    425623009: 'lateral ischaemia',
    445118002: 'left anterior fascicular block',
    253352002: 'left atrial abnormality',
    67741000119109: 'left atrial enlargement',
    446813000: 'left atrial hypertrophy',
    39732003: 'left axis deviation',
    164909002: 'left bundle branch block',
    445211001: 'left posterior fascicular block',
    164873001: 'left ventricular hypertrophy',
    370365005: 'left ventricular strain',
    251146004: 'low qrs voltages',
    251147008: 'low qrs voltages in the limb leads',
    251148003: 'low qrs voltages in the precordial leads',
    28189009: 'mobitz type 2 second degree atrioventricular block',
    54016002: 'mobitz type i wenckebach atrioventricular block',
    713423005: 'multifocal atrial tachycardia',
    164865005: 'myocardial infarction',
    164861001: 'myocardial ischemia',
    65778007: 'non-specific interatrial conduction block',
    698252002: 'nonspecific intraventricular conduction disorder',
    428750005: 'nonspecific st t abnormality',
    164867002: 'old myocardial infarction',
    10370003: 'pacing rhythm',
    251182009: 'paired ventricular premature complexes',
    282825002: 'paroxysmal atrial fibrillation',
    67198005: 'paroxysmal supraventricular tachycardia',
    425856008: 'paroxysmal ventricular tachycardia',
    164903001: 'partial atrioventricular block 2:1',
    284470004: 'premature atrial contraction',
    164884008: 'premature ventricular complexes',
    427172004: 'premature ventricular contractions',
    164947007: 'prolonged pr interval',
    111975006: 'prolonged qt interval',
    164917005: 'qwave abnormal',
    164921003: 'r wave abnormal',
    314208002: 'rapid atrial fibrillation',
    253339007: 'right atrial abnormality',
    446358003: 'right atrial hypertrophy',
    47665007: 'right axis deviation',
    59118001: 'right bundle branch block',
    89792004: 'right ventricular hypertrophy',
    55930002: 's t changes',
    49578007: 'shortened pr interval',
    427393009: 'sinus arrhythmia',
    426177001: 'sinus bradycardia',
    60423000: 'sinus node dysfunction',
    426783006: 'sinus rhythm',
    427084000: 'sinus tachycardia',
    429622005: 'st depression',
    164931005: 'st elevation',
    164930006: 'st interval abnormal',
    251168009: 'supraventricular bigeminy',
    63593006: 'supraventricular premature beats',
    426761007: 'supraventricular tachycardia',
    251139008: 'suspect arm ecg leads reversed',
    164934002: 't wave abnormal',
    59931005: 't wave inversion',
    251242005: 'tall u wave',
    266257000: 'transient ischemic attack',
    164937009: 'u wave abnormal',
    11157007: 'ventricular bigeminy',
    17338001: 'ventricular ectopic beats',
    75532003: 'ventricular escape beat',
    81898007: 'ventricular escape rhythm',
    164896001: 'ventricular fibrillation',
    111288001: 'ventricular flutter',
    266249003: 'ventricular hypertrophy',
    251266004: 'ventricular pacing pattern',
    195060002: 'ventricular pre excitation',
    164895002: 'ventricular tachycardia',
    251180001: 'ventricular trigeminy',
    195101003: 'wandering atrial pacemaker',
    74390002: 'wolff parkinson white pattern'
}
normal = [
    'NORM', 'SARRH', 'SBRAD', 'SR', 'STACH', 'Bradycardia', 'sinus arrhythmia',
    'sinus bradycardia', 'sinus rhythm', 'sinus tachycardia'
]
cd = [
    'AVB', '1AVB', '2AVB', '3AVB', 'CD', 'CLBBB', 'CRBBB', 'ILBBB', 'IRBBB',
    'IVCB', 'IVCD', 'LAFB', 'LAFB/LPFB', 'LPFB', 'LPR', 'PSVT', 'SVARR',
    'SVTAC', 'WPW', '1st degree av block', '2nd degree av block',
    'accelerated idioventricular rhythm', 'accelerated junctional rhythm',
    'Atrial pacing pattern', 'Atrial tachycardia', 'AV block',
    'Brady Tachy syndrome', 'Bundle branch block', 'Cardiac dysrhythmia',
    'complete heart block', 'complete right bundle branch block',
    'congenital incomplete atrioventricular heart block',
    'diffuse intraventricular block', 'ectopic rhythm',
    'idioventricular rhythm', 'incomplete left bundle branch block',
    'incomplete right bundle branch block', 'junctional escape',
    'junctional premature complex', 'junctional tachycardia',
    'left anterior fascicular block', 'left bundle branch block',
    'left posterior fascicular block',
    'mobitz type 2 second degree atrioventricular block',
    'mobitz type i wenckebach atrioventricular block',
    'multifocal atrial tachycardia', 'paroxysmal supraventricular tachycardia',
    'paroxysmal ventricular tachycardia', 'partial atrioventricular block 2:1',
    'prolonged pr interval', 'right bundle branch block',
    'shortened pr interval', 'sinus node dysfunction',
    'supraventricular bigeminy', 'supraventricular premature beats',
    'supraventricular tachycardia', 'ventricular ectopic beats',
    'ventricular escape beat', 'ventricular escape rhythm',
    'ventricular fibrillation', 'ventricular flutter',
    'ventricular pacing pattern', 'ventricular pre excitation',
    'ventricular tachycardia', 'ventricular trigeminy',
    'wandering atrial pacemaker', 'wolff parkinson white pattern'
]
hyp = [
    'HYP', 'ALAD', 'LAD', 'LAO/LAE', 'LVH', 'RAD', 'RHV', 'RVH', 'RAO/RAE',
    'SEHYP', 'VCLVH', 'Atrial hypertrophy', 'left atrial abnormality',
    'left atrial enlargement', 'left atrial hypertrophy',
    'left axis deviation', 'left ventricular hypertrophy',
    'left ventricular strain', 'r wave abnormal', 'right atrial abnormality',
    'right atrial hypertrophy', 'right axis deviation',
    'right ventricular hypertrophy', 'ventricular hypertrophy'
]

mi = [
    'AMI', 'ALMI', 'ASMI', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJLA',
    'INVT', 'IPLMI', 'IPMI', 'LMI', 'MI', 'PMI', 'Acute myocardial infarction',
    'Acute myocardial ischemia', 'Anterior ischemia',
    'chronic myocardial ischemia', 'inferior ischaemia',
    'inferior st segment depression', 'lateral ischaemia',
    'myocardial infarction', 'myocardial ischemia', 'old myocardial infarction'
]
sttc = [
    'ANEUR', 'DIG', 'EL', 'ISC_', 'ISCA', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCI',
    'ISCIL', 'ISCIN', 'ISCLA', 'LNGQT', 'NDT', 'NST_', 'NT_', 'STD_', 'STE_',
    'STTC', 'TAB_', 'coronary heart disease', 'electrical alternans',
    'high t voltage', 'nonspecific st t abnormality', 's t changes',
    'st depression', 'st elevation', 'st interval abnormal', 't wave abnormal',
    't wave inversion'
]
afib = [
    'AFIB', 'AFLT', 'Atrial fibrillation', 'Atrial fibrillation and flutter',
    'Atrial flutter', 'chronic atrial fibrillation',
    'paroxysmal atrial fibrillation', 'rapid atrial fibrillation'
]

other = [
    'ABQRS', 'ARAD', 'AXL', 'AXR\t', 'BIGU', 'HVOLT', 'LOWT', 'LVOLT', 'PACE',
    'PAC', 'PRC(S)', 'PVC', 'QWAVE', 'SAG', 'TRIGU', 'Abnormal QRS',
    'Atrial bigeminy', 'Blocked premature atrial contraction',
    'Brugada syndrome', 'chronic rheumatic pericarditis',
    'decreased qt interval', 'early repolarization', 'ecg artefacts',
    'fusion beats', 'heart failure', 'indeterminate cardiac axis',
    'isorhythmic dissociation', 'low qrs voltages',
    'low qrs voltages in the limb leads',
    'low qrs voltages in the precordial leads',
    'non-specific interatrial conduction block',
    'nonspecific intraventricular conduction disorder', 'pacing rhythm',
    'paired ventricular premature complexes', 'premature atrial contraction',
    'premature ventricular complexes', 'premature ventricular contractions',
    'prolonged qt interval', 'qwave abnormal',
    'suspect arm ecg leads reversed', 'tall u wave',
    'transient ischemic attack', 'u wave abnormal', 'ventricular bigeminy'
]

map1 = {
    'normal': normal,
    'cd': cd,
    'mi': mi,
    'sttc': sttc,
    'other': other,
    'afib': afib,
    'hyp': hyp
}

def get_labels(directory):
    header_files = find_header_files(directory)
    df = pd.DataFrame(columns=[
        'file_name', 'normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp'
    ])
    for i in tqdm(range(len(header_files))):
        h = header_files[i]
        set_of_labels = list()
        header = load_header(h)
        for l in header.split('\n'):
            if l.startswith('#Dx'):
                try:
                    entries = l.split(': ')[1].split(',')
                    for entry in entries:
                        set_of_labels.append(entry.strip())
                except:
                    pass
        row = pd.DataFrame(columns=[
            'file_name', 'normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp'
        ])

        row.loc[0, 'file_name'] = h

        for c in ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']:
            row[c] = 0
            for s in set_of_labels:
                s = int(s)
                if ga[s] in map1[c]:
                    row[c] = 1
        df = pd.concat([df, row]).reset_index(drop=True)
    return df


a = get_labels('/home/ubuntu/raw/ecg/WFDB_Ga')
a['patient'] = a.file_name.str.upper().apply(
    lambda x: x.split('/')[-1].split('.')[0])

a.to_csv("Ga.csv")
a = pd.read_csv("Ga.csv")
TRAIN_SPLIT_RATIO = 0.8
print(a.head(10))
unique_counts = a[['normal', 'cd', 'mi', 'sttc', 'other', 'afib',
                   'hyp']].value_counts()
index = pd.DataFrame(columns=a.columns.values)
for c_value, l in unique_counts.items():
    s = ((a['normal'] == c_value[0]) & (a['cd'] == c_value[1]) &
         (a['mi'] == c_value[2]) & (a['sttc'] == c_value[3]) &
         (a['other'] == c_value[4]) & (a['afib'] == c_value[5]) &
         (a['hyp'] == c_value[6]))

    df_sub = a[s]
    g = df_sub.sample(frac=TRAIN_SPLIT_RATIO, replace=False)
    index = pd.concat([index, g]).reset_index(drop=True)
train_patients = index['patient'].values.tolist()
a['split'] = 'val'
for i in range(a.shape[0]):
    if a.loc[i, 'patient'] in train_patients:
        a.loc[i, 'split'] = 'train'
a['label'] = -1
for i in tqdm(range(a.shape[0])):
    np.random.seed(i)
    b = a.loc[i, ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']].values
    a.loc[i, 'label'] = np.random.choice(np.flatnonzero(b == b.max()))
print(a.head(10))
print(a.label.value_counts())

a.to_csv("Ga_splits.csv")

### Generate labels for the CPSC dataset###
cpsc = {
    270492004: '1st degree av block',
    164889003: 'atrial fibrillation',
    164909002: 'left bundle branch block',
    284470004: 'premature atrial contraction',
    59118001: 'right bundle branch block',
    426783006: 'sinus rhythm',
    429622005: 'st depression',
    164931005: 'st elevation',
    164884008: 'ventricular ectopics'
}
normal = ['sinus rhythm']
cd = [
    '1st degree av block', 'atrial fibrillation', 'right bundle branch block',
    'ventricular ectopics'
]
hyp = [
    'HYP', 'ALAD', 'LAD', 'LAO/LAE', 'LVH', 'RAD', 'RHV', 'RVH', 'RAO/RAE',
    'SEHYP', 'VCLVH'
]

mi = [
    'AMI', 'ALMI', 'ASMI', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJLA',
    'INVT', 'IPLMI', 'IPMI', 'LMI', 'MI', 'PMI'
]
sttc = [
    'ANEUR', 'st depression', 'st elevation', 'DIG', 'EL', 'ISC_', 'ISCA',
    'ISCAL', 'ISCAN', 'ISCAS', 'ISCI', 'ISCIL', 'ISCIN', 'ISCLA', 'LNGQT',
    'NDT', 'NST_', 'NT_', 'STD_', 'STE_', 'STTC', 'TAB_'
]
afib = ['AFIB', 'AFLT', 'atrial fibrillation']

other = [
    'ABQRS', 'premature atrial contraction', 'ARAD', 'AXL', 'AXR\t', 'BIGU',
    'HVOLT', 'LOWT', 'LVOLT', 'PACE', 'PAC', 'PRC(S)', 'PVC', 'QWAVE', 'SAG',
    'TRIGU'
]

map1 = {
    'normal': normal,
    'cd': cd,
    'mi': mi,
    'sttc': sttc,
    'other': other,
    'afib': afib,
    'hyp': hyp
}

def get_labels(directory):
    header_files = find_header_files(directory)
    df = pd.DataFrame(columns=[
        'file_name', 'normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp'
    ])
    for i in tqdm(range(len(header_files))):
        h = header_files[i]
        set_of_labels = list()
        header = load_header(h)
        for l in header.split('\n'):
            if l.startswith('#Dx'):
                try:
                    entries = l.split(': ')[1].split(',')
                    for entry in entries:
                        set_of_labels.append(entry.strip())
                except:
                    pass
        row = pd.DataFrame(columns=[
            'file_name', 'normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp'
        ])

        row.loc[0, 'file_name'] = h

        for c in ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']:
            row[c] = 0
            for s in set_of_labels:
                s = int(s)
                if cpsc[s] in map1[c]:
                    row[c] = 1
        df = pd.concat([df, row]).reset_index(drop=True)
    return df


a = get_labels('/home/ubuntu/raw/ecg/WFDB_CPSC')
a['patient'] = a.file_name.str.upper().apply(
    lambda x: x.split('/')[-1].split('.')[0])

a.to_csv("CPSC.csv")
a = pd.read_csv("CPSC.csv")
TRAIN_SPLIT_RATIO = 0.8
print(a.head(10))
unique_counts = a[['normal', 'cd', 'mi', 'sttc', 'other', 'afib',
                   'hyp']].value_counts()
index = pd.DataFrame(columns=a.columns.values)
for c_value, l in unique_counts.items():
    s = ((a['normal'] == c_value[0]) & (a['cd'] == c_value[1]) &
         (a['mi'] == c_value[2]) & (a['sttc'] == c_value[3]) &
         (a['other'] == c_value[4]) & (a['afib'] == c_value[5]) &
         (a['hyp'] == c_value[6]))

    df_sub = a[s]
    g = df_sub.sample(frac=TRAIN_SPLIT_RATIO, replace=False)
    index = pd.concat([index, g]).reset_index(drop=True)
train_patients = index['patient'].values.tolist()
a['split'] = 'val'
for i in range(a.shape[0]):
    if a.loc[i, 'patient'] in train_patients:
        a.loc[i, 'split'] = 'train'
a['label'] = -1
for i in tqdm(range(a.shape[0])):
    np.random.seed(i)
    b = a.loc[i, ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']].values
    a.loc[i, 'label'] = np.random.choice(np.flatnonzero(b == b.max()))
print(a.head(10))
print(a.label.value_counts())

a.to_csv("CPSC_splits.csv")

### Generate labels for the Chapman-Shaoxing dataset###

cs = {
    251199005: 'Counterclockwise cardiac rotation',
    270492004: '1AVB',
    195042002: '2AVB2',
    251173003: 'ABI',
    164890007: 'AF',
    164889003: 'AFIB',
    39732003: 'ALS',
    284470004: 'APB',
    164917005: 'AQW',
    47665007: 'ARS',
    713422000: 'AT',
    233917008: 'AVB',
    251166008: 'AVNRT',
    27885002: 'CAVB',
    164909002: 'CLBBB',
    251198002: 'CR',
    428417006: 'ERV',
    164942001: 'FQRS',
    54016002: 'IIAVBI',
    698252002: 'IVB',
    426995002: 'JEB',
    251164006: 'JPT',
    164873001: 'LVH',
    55827005: 'LVHV',
    251146004: 'PRWP',
    164865005: 'MILW',
    59118001: 'nonspecific BBB',
    164947007: 'PRIE',
    111975006: 'PTW',
    164912004: 'PWC',
    446358003: 'RAH',
    67751000119106: 'RAHV',
    233897008: 'RAVC',
    89792004: 'RVH',
    17366009: 'SAAWR',
    426177001: 'SB',
    426783006: 'SR',
    427084000: 'ST',
    429622005: 'STDD',
    164930006: 'STE',
    428750005: 'STTC',
    164931005: 'STTU',
    426761007: 'SVT',
    164934002: 'TTW',
    59931005: 'TWO',
    164937009: 'UW',
    11157007: 'VB',
    75532003: 'VEB',
    251180001: 'VET',
    13640000: 'VFW',
    17338001: 'VPB',
    195060002: 'VPE',
    195101003: 'WAVN',
    74390002: 'WPW'
}
normal = ['SB', 'SR', 'ST']
cd = [
    '1AVB', '2AVB2', 'AVB', 'AVNRT', 'AT', 'CAVB', 'CLBBB', 'IIAVBI', 'IVB',
    'JEB', 'JPT', 'Nonspecific BBB', 'PRIE', 'PRWP', 'PWC', 'SAAWR', 'SVT',
    'VEB', 'VET', 'VPB', 'VPE', 'WAVN', 'WPW'
]
hyp = ['ALS', 'ARS', 'CR', 'LVH', 'LVHV', 'RAH', 'RAVC', 'RVH']

mi = ['MILW']
sttc = ['STDD', 'STE', 'STTC', 'STTU', 'TTW', 'TWO']
afib = ['AF', 'AFIB']

other = [
    'ABI', 'APB', 'AQW', 'ERV', 'FQRS', 'LVQRSCL', 'LVQRSLL', 'PTW', 'UW', 'VB'
]
map1 = {
    'normal': normal,
    'cd': cd,
    'mi': mi,
    'sttc': sttc,
    'other': other,
    'afib': afib,
    'hyp': hyp
}

def get_labels(directory):
    header_files = find_header_files(directory)
    df = pd.DataFrame(columns=[
        'file_name', 'normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp'
    ])
    for i in tqdm(range(len(header_files))):
        h = header_files[i]
        set_of_labels = list()
        header = load_header(h)
        for l in header.split('\n'):
            if l.startswith('#Dx'):
                try:
                    entries = l.split(': ')[1].split(',')
                    for entry in entries:
                        set_of_labels.append(entry.strip())
                except:
                    pass
        row = pd.DataFrame(columns=[
            'file_name', 'normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp'
        ])

        row.loc[0, 'file_name'] = h

        for c in ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']:
            row[c] = 0
            for s in set_of_labels:
                s = int(s)
                if cs[s] in map1[c]:
                    row[c] = 1
        df = pd.concat([df, row]).reset_index(drop=True)

    return df


a = get_labels('/home/ubuntu/raw/ecg/WFDB_ChapmanShaoxing')
a['patient'] = a.file_name.str.upper().apply(
    lambda x: x.split('/')[-1].split('.')[0])

a.to_csv("ChapmanShaoxing.csv")
a = pd.read_csv("ChapmanShaoxing.csv")
TRAIN_SPLIT_RATIO = 0.8
print(a.head(10))
unique_counts = a[['normal', 'cd', 'mi', 'sttc', 'other', 'afib',
                   'hyp']].value_counts()
index = pd.DataFrame(columns=a.columns.values)
for c_value, l in unique_counts.items():
    s = ((a['normal'] == c_value[0]) & (a['cd'] == c_value[1]) &
         (a['mi'] == c_value[2]) & (a['sttc'] == c_value[3]) &
         (a['other'] == c_value[4]) & (a['afib'] == c_value[5]) &
         (a['hyp'] == c_value[6]))
    df_sub = a[s]
    g = df_sub.sample(frac=TRAIN_SPLIT_RATIO, replace=False)
    index = pd.concat([index, g]).reset_index(drop=True)
train_patients = index['patient'].values.tolist()
a['split'] = 'val'
for i in range(a.shape[0]):
    if a.loc[i, 'patient'] in train_patients:
        a.loc[i, 'split'] = 'train'
a['label'] = -1
for i in tqdm(range(a.shape[0])):
    np.random.seed(i)
    b = a.loc[i, ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']].values
    a.loc[i, 'label'] = np.random.choice(np.flatnonzero(b == b.max()))
print(a.head(10))
print(a.label.value_counts())

a.to_csv("ChapmanShaoxing_splits.csv")
